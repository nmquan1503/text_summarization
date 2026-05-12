import config
from modeling.models.seq2seq import Seq2Seq, Seq2SeqConfig
from modeling.models.causal_lm import CausalLM, CausalLMConfig
from selective_attention import models

def auto_model():
    if config.TYPE == "seq2seq":
        return Seq2Seq(Seq2SeqConfig(
            vocab_size=config.VOCAB_SIZE,
            pad_token_id=config.PAD_ID,
            bos_token_id=config.BOS_ID,
            eos_token_id=config.EOS_ID,
            model_dim=config.MODEL_DIM,
            state_dim=config.STATE_DIM,
            conv_kernel=config.CONV_KERNEL,
            head_dim=config.HEAD_DIM,
            num_groups=config.NUM_GROUPS,
            chunk_size=config.CHUNK_SIZE,
            num_layers=config.NUM_LAYERS,
            device="cuda"
        )).to("cuda")
    elif config.TYPE == "causal_lm":
        # return CausalLM(CausalLMConfig(
        #     vocab_size=config.VOCAB_SIZE,
        #     pad_token_id=config.PAD_ID,
        #     bos_token_id=config.BOS_ID,
        #     eos_token_id=config.EOS_ID,
        #     model_dim=config.MODEL_DIM,
        #     state_dim=config.STATE_DIM,
        #     conv_kernel=config.CONV_KERNEL,
        #     head_dim=config.HEAD_DIM,
        #     num_groups=config.NUM_GROUPS,
        #     chunk_size=config.CHUNK_SIZE,
        #     num_layers=config.NUM_LAYERS,
        #     device="cuda"
        # )).to("cuda")
        return models.CausalLM(models.CausalLMConfig(
            vocab_size=config.VOCAB_SIZE,
            model_dim=config.MODEL_DIM,
            head_dim=config.HEAD_DIM,
            ssm_state_dim=config.SSM_STATE_DIM,
            ssm_conv_kernel_size=config.SSM_CONV_KERNEL_SIZE,
            ssm_num_groups=config.SSM_NUM_GROUPS,
            ssm_chunk_size=config.SSM_CHUNK_SIZE,
            mlconv_radius=config.MLCONV_RADIUS,
            num_layers=config.NUM_LAYERS
        ))