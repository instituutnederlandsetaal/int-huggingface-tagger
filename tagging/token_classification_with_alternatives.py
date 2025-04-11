from transformers import AutoTokenizer, AutoModelForTokenClassification
from tokenizers import pre_tokenizers
from transformers import pipeline
from transformers.pipelines import PIPELINE_REGISTRY
from transformers.pipelines.token_classification import TokenClassificationPipeline, AggregationStrategy
import re
import sys
import os
from typing import Callable, Mapping, List

from nltk.tokenize import sent_tokenize, word_tokenize
from dataclasses import dataclass

# cf https://huggingface.co/docs/transformers/add_new_pipeline


# override token classification pipeline to give user access to list of scored alternative labels
# caution: will probably break on huggingface updates, works with transformers-4.34.0
# a typical test is enhance_bab.py

class TokenClassificationWithAlternativesPipeline(TokenClassificationPipeline):
    def aggregate(self, pre_entities: List[dict], aggregation_strategy: AggregationStrategy) -> List[dict]:
        if aggregation_strategy in {AggregationStrategy.NONE, AggregationStrategy.SIMPLE}:
            entities = []
            for pre_entity in pre_entities:
                entity_idx = pre_entity["scores"].argmax()
                scores = pre_entity["scores"] # OOPS
                # print(f"{len(pre_entities)}:{len(scores)}:{type(scores)}") # OOPS
                alternatives = sorted([(self.model.config.id2label[i], scores[i]) for i in range(0,len(scores))],key=lambda x: -1 * x[1])
                chosen =  self.model.config.id2label[entity_idx]
                word =  pre_entity["word"]
                # print(word + " : " + chosen + " or choose between: " +  str(alternatives))

                score = pre_entity["scores"][entity_idx]
                entity = {
                    "entity": self.model.config.id2label[entity_idx],
                    "score": score,
                    "index": pre_entity["index"],
                    "word": pre_entity["word"],
                    "start": pre_entity["start"],
                    "end": pre_entity["end"],
                    "alternatives" : alternatives
                }
                entities.append(entity)
        else:
            entities = self.aggregate_words(pre_entities, aggregation_strategy)

        if aggregation_strategy == AggregationStrategy.NONE:
            return entities
        return self.group_entities(entities)


def register():
  PIPELINE_REGISTRY.register_pipeline(
     "token-classification-with-alternatives",
      pipeline_class=TokenClassificationWithAlternativesPipeline,
      pt_model=AutoModelForTokenClassification,
  )

