from transformers import Text2TextGenerationPipeline
from transformers.tokenization_utils_base import TruncationStrategy


class EncoderDecoderText2TextGenerationPipeline(Text2TextGenerationPipeline):
    """Text2Text pipeline for encoder-decoders that accepts a different tokenizer for the decoder and initial prompts.
    
    When `model` is an encoder-decoder, each input of the pipeline must be a dictionary with keys 'condition' (input)
    of the encoder and 'prompt' (input of the decoder).
    """
    def __init__(self, *args, decoder_tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.decoder_tokenizer = decoder_tokenizer

    def _parse_and_tokenize(self, *args, truncation, tokenizer):
        prefix = self.model.config.prefix if self.model.config.prefix is not None else ""
        if isinstance(args[0], list):
            if tokenizer.pad_token_id is None:
                raise ValueError("Please make sure that the tokenizer has a pad_token_id when using a batch input")
            args = ([prefix + arg for arg in args[0]],)
            padding = True

        elif isinstance(args[0], str):
            args = (prefix + args[0],)
            padding = False
        else:
            raise ValueError(
                f" `args[0]`: {args[0]} have the wrong format. The should be either of type `str` or type `list`"
            )
        inputs = tokenizer(*args, padding=padding, truncation=truncation, return_tensors=self.framework)
        # This is produced by tokenizers but is an invalid generate kwargs
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]
        return inputs

    def preprocess(self, inputs, truncation=TruncationStrategy.DO_NOT_TRUNCATE, **kwargs):
        if not self.model.config.is_encoder_decoder:
            inputs = self._parse_and_tokenize(inputs, truncation=truncation, tokenizer=self.tokenizer, **kwargs)
            return inputs
        else:
            assert isinstance(inputs, dict), (
                'Only a dictionary is accepted as inputs'
            )
            decoder_inputs = self._parse_and_tokenize(
                inputs['prompt'], truncation=truncation, tokenizer=self.decoder_tokenizer, **kwargs
            )
            inputs = self._parse_and_tokenize(
                inputs['condition'], truncation=truncation, tokenizer=self.tokenizer, **kwargs
            )
            if decoder_inputs['input_ids'].shape[-1] > 0:
                inputs['decoder_input_ids'] = decoder_inputs['input_ids']
                inputs['decoder_attention_mask'] = decoder_inputs['attention_mask']
            return inputs
