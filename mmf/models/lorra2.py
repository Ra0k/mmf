# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import sys

from mmf.common.registry import registry
from mmf.models.pythia import Pythia


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
                
                feature_embeddings.append(encoded_feature)
                feature_attentions.append(attention)
                #feature_attentions.append(attention.squeeze(-1))

        # Concatenate all features embeddings and return along with attention
        # feature_embedding_total = torch.cat(feature_embeddings, dim=1)
        return feature_embeddings, feature_attentions

    def forward(self, sample_list):
        print(sample_list.image_info_0, file=sys.stderr)
        
        exit()
        
        sample_list.text = self.word_embedding(sample_list.text)
        text_embedding_total = self.process_text_embedding(sample_list)

        image_embeddings, attentions = self.process_feature_embedding(
            "image", sample_list, text_embedding_total
        )
        
        #print(len(image_embeddings), file=sys.stderr)
        #print(len(attentions), file=sys.stderr)

        context_embeddings, attentions = self.process_feature_embedding(
            "context", sample_list, text_embedding_total, ["order_vectors"]
        )
        
        #print(len(context_embeddings), file=sys.stderr)
        #print(len(attentions), file=sys.stderr)
        
        #exit()

        if self.inter_model is not None:
            image_embedding_total = self.inter_model(image_embedding_total)

        joint_embedding = self.combine_embeddings(
            ["image", "text"],
            [image_embedding_total, text_embedding_total, context_embedding_total],
        )

        scores = self.calculate_logits(joint_embedding)

        return {"scores": scores}
