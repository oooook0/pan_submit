
Submit to tira via:

```
tira-cli code-submission \
	--mount-hf-model google/gemma-1.1-2b-it bert-base-uncased meta-llama/Llama-2-7b-chat-hf \
	--path . \
	--task generative-ai-authorship-verification-panclef-2025 \
	--dataset pan25-generative-ai-detection-smoke-test-20250428-training \
	--command 'python3 run.py -i $inputDataset -o $outputDir'
```
