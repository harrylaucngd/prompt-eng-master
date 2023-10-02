from transformers import pipeline

class DecomposedModel:
    def __init__(self, model_name):
        self.generator = pipeline("text-generation", model=model_name)

    def generate_text(self, prompt, target_label, max_length=50, num_return_sequences=1):
        output = self.generator(
            prompt,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            label=target_label,
            do_sample=True,
        )

        # 返回生成的文本
        return [result["generated_text"] for result in output]