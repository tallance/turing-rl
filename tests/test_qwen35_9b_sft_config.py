import os, yaml
CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "training", "sft", "configs", "qwen35_9b_lora.yaml")


def test_qwen35_9b_lora_config():
    with open(CFG) as f:
        c = yaml.safe_load(f)
    assert c["lora_r"] == 64 and c["lora_alpha"] == 128 and c["lora_dropout"] == 0.05
    assert c["use_qlora"] is False          # bf16, paper Table 5
    assert c["num_epochs"] == 3
