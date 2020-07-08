# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import sys

from mmf.common.registry import registry
from mmf.models.pythia import Pythia
from torch.nn.functional import softmax


@registry.register_model("lorra2")
class LoRRA2(Pythia):
    def __init__(self, config):
        super().__init__(config)

    @classmethod
    def config_path(cls):
        return "configs/models/lorra2/defaults.yaml"

    def build(self):
        self._init_text_embeddings("text")
        # For LoRRA context feature and text embeddings would be identity
        # but to keep a unified API, we will init them also
        # and we need to build them first before building pythia's other
        # modules as some of the modules require context attributes to be set
        self._init_text_embeddings("context")
        self._init_feature_encoders("context")
        self._init_feature_embeddings("context")
        super().build()

    def get_optimizer_parameters(self, config):
        params = super().get_optimizer_parameters(config)
        params += [
            {"params": self.context_feature_embeddings_list.parameters()},
            {"params": self.context_embeddings.parameters()},
            {"params": self.context_feature_encoders.parameters()},
        ]

        return params

    def _get_classifier_input_dim(self):
        # Now, the classifier's input will be cat of image and context based
        # features
        return 2 * super()._get_classifier_input_dim()
    
    def process_feature_embedding(
        self, attr, sample_list, text_embedding_total, extra=None, batch_size_t=None
    ):
        if extra is None:
            extra = []
        total_embeddings = []
        feature_embeddings = []
        feature_attentions = []
        features = []
        batch_size_t = (
            sample_list.get_batch_size() if batch_size_t is None else batch_size_t
        )

        # Convert list of keys to the actual values
        extra = sample_list.get_fields(extra)

        feature_idx = 0

        # Get all of the features, which are in the form, "image_feature_0"
        # "image_feature_1" ...
        while True:
            feature = getattr(sample_list, f"{attr}_feature_{feature_idx:d}", None)
            if feature is None:
                break
            feature_idx += 1
            feature = feature[:batch_size_t]
            features.append(feature)

        feature_encoders = getattr(self, attr + "_feature_encoders")
        # Each feature should have a separate image feature encoders
        assert len(features) == len(feature_encoders), (
            "Number of feature encoders, {} are not equal "
            "to number of features, {}.".format(len(feature_encoders), len(features))
        )

        # Now, iterate to get final attended image features
        for i, feature in enumerate(features):
            # Get info related to the current feature. info is generally
            # in key of format "image_info_0" for 0th feature
            feature_info = getattr(sample_list, f"{attr}_info_{i:d}", {})
            # For Pythia, we need max_features to mask attention
            feature_dim = getattr(feature_info, "max_features", None)
            if feature_dim is not None:
                feature_dim = feature_dim[:batch_size_t]

            # Attribute in which encoders are saved, for "image" it
            # will be "image_feature_encoders", other example is
            # "context_feature_encoders"
            encoders_attr = attr + "_feature_encoders"
            feature_encoder = getattr(self, encoders_attr)[i]

            # Encode the features
            encoded_feature = feature_encoder(feature)

            # Get all of the feature embeddings
            list_attr = attr + "_feature_embeddings_list"
            feature_embedding_models = getattr(self, list_attr)[i]

            # Forward through these embeddings one by one
            for feature_embedding_model in feature_embedding_models:
                inp = (encoded_feature, text_embedding_total, feature_dim, extra)

                embedding, attention = feature_embedding_model(*inp)

                #total = torch.sum(encoded_feature * attention, dim=1).view(embedding.size())

                #print('extra ', extra.keys(), len(total_embeddings), embedding.shape, file=sys.stderr)
                total_embeddings.append(embedding)
                feature_embeddings.append(encoded_feature)
                feature_attentions.append(attention)

        return feature_embeddings, feature_attentions, total_embeddings
    
    def bb_intersection_over_union(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        interArea = max(0, xB - xA + 0.00001) * max(0, yB - yA + 0.00001)
        if interArea > 0:
            True
        else:
            False
    
    def matching(self, batch_image_bboxes, batch_image_attentions, batch_context_bboxes, batch_context_attentions):
        image_plus_attention = torch.zeros(batch_image_attentions.size())
        context_plus_attention = torch.zeros(batch_context_attentions.size())
        
        for batch_idx in range(len(batch_image_bboxes)):
            image_bboxes = batch_image_bboxes[batch_idx]
            context_bboxes = batch_context_bboxes[batch_idx]
            image_attentions = batch_image_attentions[batch_idx]
            context_attentions = batch_context_attentions[batch_idx]
            for idx, (image_bbox, image_attention) in enumerate(zip(image_bboxes, image_attentions)):
                if image_attention == 0: break # skip zeros generated by padding
                for cdx, (context_bbox, context_attention) in enumerate(zip(context_bboxes,context_attentions)):
                    if context_attention == 0: break # skip zeros generated by padding
                    is_intersect = self.bb_intersection_over_union(image_bbox, context_bbox)
                    if is_intersect:
                        if image_attention > context_plus_attention[batch_idx][cdx]:
                            context_plus_attention[batch_idx][cdx] = image_attention
                        if context_attention > image_plus_attention[batch_idx][idx]:
                            image_plus_attention[batch_idx][idx] = context_attention
        return image_plus_attention, context_plus_attention
    

    def forward(self, sample_list):
        
        # normalize bbox information
        for idx, sample in enumerate(sample_list.image_info_0.bbox):
            w = sample_list.image_info_0.image_width[idx]
            h = sample_list.image_info_0.image_height[idx]
            for idy, bbox in enumerate(sample):
                sample_list.image_info_0.bbox[idx][idy][0] = min(bbox[0]/w, 1.0)
                sample_list.image_info_0.bbox[idx][idy][2] = min(bbox[2]/w, 1.0)
                sample_list.image_info_0.bbox[idx][idy][1] = min(bbox[1]/h, 1.0)
                sample_list.image_info_0.bbox[idx][idy][3] = min(bbox[3]/h, 1.0)
                

        sample_list.text = self.word_embedding(sample_list.text)
        text_embedding_total = self.process_text_embedding(sample_list)

        image_embeddings, image_attentions, total_image_embeddings = self.process_feature_embedding(
            "image", sample_list, text_embedding_total
        )

        context_embeddings, context_attentions, total_context_embeddings = self.process_feature_embedding(
            "context", sample_list, text_embedding_total, ["order_vectors"]
        )
        

        #import time
        #start_time = time.time()
        
        image_plus_attention, context_plus_attention = self.matching(
            sample_list.image_info_0.bbox, image_attentions[0],
            sample_list.ocr_bbox_coordinates, context_attentions[0]
        )
        
        #print("--- %s seconds ---" % (time.time() - start_time), file=sys.stderr)
        
        #print(image_plus_attention.shape, image_attentions[0].shape, file=sys.stderr)
        #print(context_plus_attention.shape,  context_attentions[0].shape, file=sys.stderr)
        
        image_attentions[0] = softmax(image_attentions[0] + (image_attentions[0] * image_plus_attention), dim=1)
        context_attentions[0] = softmax(context_attentions[0] + (context_attentions[0] * context_plus_attention), dim=1)
        
        feature_1_total = torch.sum(image_embeddings[0] * image_attentions[0], dim=1).view(
            image_embeddings[0].shape[0], image_embeddings[0].shape[2]
        )
        
        #print(sample_list.image_info_0.bbox[0], image_attentions[0][0],
        #    sample_list.ocr_bbox_coordinates[0], context_attentions[0][0], file=sys.stderr)
        

        #print(len(context_embeddings), file=sys.stderr)
        #print(image_plus_attention, context_plus_attention, file=sys.stderr)
        
        image_embedding_total = feature_1_total
        
        if len(total_image_embeddings) == 2:
            #feature_2_total = torch.sum(
            #    image_embeddings[1] * image_attentions[1], dim=1
            #).view(image_embeddings[1].shape[0],image_embeddings[1].shape[2])
            feature_2_total = total_image_embeddings[1]
            
            image_embedding_total = torch.cat([feature_1_total, feature_2_total], dim=1)
        
        context_embedding_total = torch.sum(context_embeddings[0] * context_attentions[0], dim=1).view(
            context_embeddings[0].shape[0], context_embeddings[0].shape[2]
        )
        
        context_embedding_total = torch.cat([context_embedding_total, total_context_embeddings[0][:,300:]], dim=1)
            
        if self.inter_model is not None:
            image_embedding_total = self.inter_model(image_embedding_total)

        #print(image_embedding_total.shape, text_embedding_total.shape, context_embedding_total.shape, file=sys.stderr)
        #exit()
            
        joint_embedding = self.combine_embeddings(
            ["image", "text"],
            [image_embedding_total, text_embedding_total, context_embedding_total],
        )

        scores = self.calculate_logits(joint_embedding)

        return {"scores": scores}
