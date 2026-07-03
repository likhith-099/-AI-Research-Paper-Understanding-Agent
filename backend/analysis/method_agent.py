from analysis.section_service import generate_evidence_based_section


def generate_method_explanation(retriever):
    return generate_evidence_based_section(
        retriever,
        section_name="method",
        query_text="""
        methodology
        method
        approach
        architecture
        model design
        training procedure
        training pipeline
        optimization
        encoder
        backbone
        feature extraction
        representation learning
        pretraining
        contrastive learning
        self-supervised learning
        MoCo
        SimCLR
        SwAV
        Barlow Twins
        SimSiam
        ViT
        ViT-B/16
        MAE
        LoRA
        QLoRA
        Swin-T
        """,
        max_tokens=750,
        prompt="""
Explain the methodology used in the paper.

Instructions:
- Explain the core method in simple technical language.
- Describe the architecture, framework, or algorithm.
- Explain how the system works step-by-step.
- Mention training strategy if available.
- Mention important components, encoders, backbones, modules, or pipelines.
- Do not invent information.
- If some details are missing, only explain what is present.
- Write 1 concise paragraph of 5-8 sentences.
- Avoid bullet points.
""".strip(),
    )
