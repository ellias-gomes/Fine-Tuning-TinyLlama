# Fine-Tuning TinyLlama (Colab Notebook)

Notebook Jupyter / Google Colab com experimentos práticos de **fine-tuning com LoRA**, **treinamento de RNN do zero em PyTorch** e **construção de um assistente conversacional local** com raciocínio passo a passo (`<think>`), memória de conversa e busca na internet via DuckDuckGo.

> **Nota:** Embora o nome do projeto mencione *TinyLlama*, as células principais utilizam os modelos abertos **`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`** (para fine-tuning e inferência com LoRA) e **`Qwen/Qwen2.5-1.5B-Instruct`** (para raciocínio, busca web e memória).

## O que o notebook faz

O arquivo [`treinar_tinyllama.ipynb`](treinar_tinyllama.ipynb) está organizado em etapas progressivas:

1. **Fine-Tuning com LoRA (`PEFT` + `Transformers`)**
   - Carrega o modelo `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` e aplica adaptadores LoRA (`r=8`, `lora_alpha=16`, módulos `q_proj` e `v_proj`).
   - Lê frases de treinamento de um arquivo local `conhecimento.txt` usando um `Dataset` customizado e treina por 50 épocas com `Trainer`.
   - Salva o adaptador treinado em `./meu_chatbot_tinyllama` (e checkpoints em `./resultados_lora`).

2. **Preparação de Dataset em Português**
   - Baixa 5.000 exemplos de `dominguesm/alpaca-data-pt-br` via `datasets`.
   - Formata os pares no padrão `<pergunta> ... <resposta> ...` e salva em `conhecimento_gigante.txt`.

3. **Chatbot Experimental com RNN (PyTorch puro)**
   - Implementa a arquitetura `ChatbotPro` (`Embedding` + `RNN` + `Linear`) treinada palavra por palavra sobre `conhecimento_gigante.txt`, salvando pesos em `chatbot_pesos.pth`.
   - Inclui loop interativo no terminal com o comando `/ensinar` para adicionar novos pares de pergunta/resposta e retreinar.

4. **Inferência Local com Adaptador LoRA**
   - Carrega o modelo base e acopla `./meu_chatbot_tinyllama` via `PeftModel.from_pretrained` para chat em CPU.

5. **Assistente com Raciocínio (`<think>`), Busca na Web e Memória**
   - Utiliza `Qwen/Qwen2.5-1.5B-Instruct` com `apply_chat_template`.
   - Separa o raciocínio interno (`<think>...</think>`) da resposta final.
   - Integra busca em tempo real com DuckDuckGo (`ddgs` / `duckduckgo_search`) quando o modelo emite a tag `[PESQUISAR]`.
   - Mantém uma janela deslizante de memória com as últimas 6 mensagens da conversa.

## Estrutura do Repositório

- [`treinar_tinyllama.ipynb`](treinar_tinyllama.ipynb): Notebook principal com todas as células de treinamento, inferência e agente.
- [`.gitignore`](.gitignore): Ignora arquivos temporários, checkpoints de treinamento (`resultados_lora/`, `meu_chatbot_tinyllama/`), pesos (`*.pth`) e datasets gerados (`conhecimento*.txt`).

## Dependências Principais

Para executar no Google Colab ou ambiente local com Python:

```bash
pip install torch transformers peft datasets duckduckgo-search
```

## Dados Sensíveis

Nenhum token, chave de API ou credencial foi encontrado no notebook; todos os modelos e datasets utilizados são públicos no Hugging Face e as buscas utilizam o DuckDuckGo sem necessidade de chave.
