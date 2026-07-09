# EHT MLOps Chatbot RAG Pipeline

Dashboard interativo para demonstrar uma suíte MLOps/LLMOps inspirada no Event Horizon Telescope, com três pipelines Airflow, model registry, monitoramento de drift e chatbot RAG.

GitHub Pages:

```text
https://arteweyl.github.io/EHT-MLOps-Chatbot-RAG-Pipeline/
```

Este projeto separa e estende a parte de MLOps do processamento do telescópio EHT (Event Horizon Telescope) para modelar M87* em uma plataforma própria, aplicando as **boas práticas da palestra da Python Norte 2026** baseada em 3 pipelines fundamentais.

## 🏗️ Estrutura dos 3 Pipelines

A arquitetura do projeto implementa três DAGs do Airflow usando a moderna **TaskFlow API** com tipagem estática e idempotência via particionamento lógico:

1. **DAG 01: Pipeline de Otimização & Reconstrução (`01_eht_image_reconstruction`)**
   - Ingestão determinística de dados VLBI.
   - Validação estrutural de schema dos dados.
   - Autocalibração circular de fase complexa (estatística circular).
   - Reconstrução da imagem do céu via otimização RML (com gradientes analíticos exatos).
   - Avaliação estrutural por template matching em biblioteca GRMHD.
   - **Quality Gate (`BranchPythonOperator`)**: Promove automaticamente o modelo para `PRODUCTION` (Champion) se passar nas faixas físicas de massa ($5.0 \times 10^9 M_\odot$ a $8.0 \times 10^9 M_\odot$) e fidelidade NCC ($\ge 70\%$). Caso contrário, rejeita.

2. **DAG 02: Pipeline de Monitoramento de Drift Instrumental (`02_eht_drift_monitoring`)**
   - Carrega distribuições ideais de visibilidade como calibrador de referência.
   - Extrai observações diárias afetadas por instabilidade e ruído atmosférico.
   - Calcula o **PSI (Population Stability Index)** das amplitudes de rádio.
   - **Drift Gate**: Se o desvio acumula instabilidade suficiente ($\text{PSI} \ge 0.20$), o **`TriggerDagRunOperator`** é acionado automaticamente, forçando a re-execução (retreino/re-calibração) do pipeline de reconstrução principal (DAG 01).

3. **DAG 03: Pipeline RAG / LLMOps (`03_eht_catalog_rag_refresh`)**
   - Extrai metadados do modelo champion ativo no Model Registry.
   - Fragmenta os resultados em chunks informativos (Massa, Spin, Fidelidade).
   - Gera embeddings vetoriais determinísticos locais (sin-based cross-platform).
   - Atualiza o banco de dados vetorial de busca (`eht_rag_index.json`).
   - Valida o frescor do índice (Freshness validation).

---

## 📁 Layout do Projeto

```text
eht_mlops_airflow/
├── dags/
│   ├── 01_eht_image_reconstruction.py
│   ├── 02_eht_drift_monitoring.py
│   └── 03_eht_catalog_rag_refresh.py
├── src/
│   ├── __init__.py
│   ├── simulator.py        # Simulação VLBI & ground truth
│   ├── calibrator.py       # Autocalibração circular & ruído
│   ├── reconstruction.py   # RML com gradientes analíticos
│   ├── evaluator.py        # Biblioteca GRMHD & NCC metrics
│   ├── mlops_registry.py   # Logs de runs e promoção
│   └── rag_index.py        # Chunks, embeddings & vector index
├── index.html              # Dashboard interativo web
├── style.css               # Estilos premium em dark mode
├── script.js               # Animações de DAG e Chatbot RAG
├── requirements.txt        # Dependências python
└── run_all_pipelines_local.py # Executor local sequencial
```

---

## 💻 Como Executar

### 1. Iniciar o Ollama com o Qwen
Para rodar o Chatbot em tempo real conectado a um LLM local:
```bash
# Baixar e rodar o Qwen 1.5B/2B no terminal
ollama run qwen2:1.5b
```

### 2. Rodar os pipelines em Python
Para simular a orquestração do Airflow e gerar os logs que alimentam o site:
```bash
python3 run_all_pipelines_local.py
```
Isso executará a reconstrução principal, simulará um drift de calibração que dispara o retreino automático e atualizará o RAG Index.

### 3. Abrir o Dashboard Interativo
Abra o arquivo `index.html` em seu navegador para explorar os pipelines:
- **Painel de DAGs**: Acompanhe o fluxo de tarefas acendendo em verde e amarelo à medida que executam, com mini-visualizadores dinâmicos no canvas.
- **Simulador de Ruído**: Altere a severidade atmosférica no slider e ative a DAG 02 para calcular o PSI e disparar o retreino automático.
- **Chatbot Científico RAG**: Converse com o catálogo na terceira aba! Se o Ollama estiver ligado, ele responderá usando o modelo `qwen2:1.5b` alimentado com o contexto do RAG Index. Se estiver desligado, o chat exibirá as respostas locais de template de forma transparente.
