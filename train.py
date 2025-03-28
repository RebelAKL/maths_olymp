from data.kaggle_downloader import download_datasets
from model.vllm_engine import MathLLMEngine
from memory.lemma_bank import LemmaMemory
from training.trainer import MathTrainer

def main():
    download_datasets()

    llm_engine = MathLLMEngine()
    lemma_memory = LemmaMemory()
    trainer = MathTrainer(llm_engine, lemma_memory)
    
    datasets = ['MATH', 'gsm8k', 'theoremqa']
    for dataset in datasets:
        trainer.train(dataset)

    llm_engine.llm.save("trained_model")

if __name__ == "__main__":
    main()