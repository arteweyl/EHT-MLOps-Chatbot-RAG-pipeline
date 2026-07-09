// EHT MLOps Airflow Suite - Interactive Logic
document.addEventListener("DOMContentLoaded", () => {
    // State management
    let activeDag = "reconstruct"; // "reconstruct", "drift", "rag"
    let runningPipeline = null; // null or active pipeline run ID
    let modelRegistry = []; // Model history list
    let indexedDocs = []; // Document chunks database for RAG chatbot
    let driftSeverity = 2; // 1 = Low (PSI 0.05), 2 = Moderate (PSI 0.12), 3 = Severe (PSI 2.37)
    
    // DOM bindings
    const dagTabs = document.querySelectorAll(".dag-tab-btn");
    const dagViews = document.querySelectorAll(".dag-view");
    const activeDagDisplay = document.getElementById("active-dag-display");
    const triggerBtn = document.getElementById("btn-trigger-active-dag");
    
    const driftCard = document.getElementById("drift-sim-card");
    const driftSlider = document.getElementById("drift-severity-slider");
    const driftValDisplay = document.getElementById("drift-val-display");
    
    const ragCard = document.getElementById("rag-sim-card");
    const ragLastRun = document.getElementById("rag-last-run-id");
    const ragDocCount = document.getElementById("rag-document-count");
    
    const championFidelity = document.getElementById("champion-fidelity");
    const championMass = document.getElementById("champion-mass");
    const championSpin = document.getElementById("champion-spin");
    const runsTableBody = document.getElementById("runs-table-body");
    const emptyRunsRow = document.getElementById("empty-runs-row");
    
    const chatInput = document.getElementById("chat-input-text");
    const chatSendBtn = document.getElementById("btn-send-chat");
    const chatMessages = document.getElementById("chat-messages");
    const retrievedText = document.getElementById("retrieved-text");
    
    const detailsTitle = document.getElementById("node-title");
    const detailsStatusBadge = document.getElementById("node-status-badge");
    const detailsDesc = document.getElementById("node-desc");
    const plotArea = document.getElementById("node-plot-area");
    
    // Pre-populate Registry with initial reference from local JSON file
    loadInitialData();
    
    // Initialize tabs switching
    dagTabs.forEach(tab => {
        tab.addEventListener("click", () => {
            if (runningPipeline) return; // Block switching while running
            
            dagTabs.forEach(t => t.classList.remove("active"));
            dagViews.forEach(v => v.classList.remove("active"));
            
            tab.classList.add("active");
            activeDag = tab.dataset.dag;
            document.getElementById(`dag-${activeDag}`).classList.add("active");
            
            // Update display
            const dagIds = {
                reconstruct: "01_eht_image_reconstruction",
                drift: "02_eht_drift_monitoring",
                rag: "03_eht_catalog_rag_refresh"
            };
            activeDagDisplay.textContent = dagIds[activeDag];
            
            // Adjust card disabling
            if (activeDag === "drift") {
                driftCard.classList.remove("disabled");
                ragCard.classList.add("disabled");
            } else if (activeDag === "rag") {
                driftCard.classList.add("disabled");
                ragCard.classList.remove("disabled");
            } else {
                driftCard.classList.add("disabled");
                ragCard.classList.add("disabled");
            }
            
            resetDagNodesVisuals();
            showDagWelcomeDetails(activeDag);
        });
    });

    // Slider listener
    driftSlider.addEventListener("input", (e) => {
        driftSeverity = parseInt(e.target.value);
        const labels = {
            1: "Baixo (PSI ~0.04 - Sem Drift)",
            2: "Moderado (PSI ~0.12 - Atenção)",
            3: "Crítico (PSI ~2.37 - Drift Forte!)"
        };
        driftValDisplay.textContent = labels[driftSeverity];
    });

    // Trigger Dag Run handler
    triggerBtn.addEventListener("click", () => {
        if (runningPipeline) return;
        
        if (activeDag === "reconstruct") {
            runReconstructionPipeline();
        } else if (activeDag === "drift") {
            runDriftMonitoringPipeline();
        } else if (activeDag === "rag") {
            runRagRefreshPipeline();
        }
    });

    // Ollama configuration elements
    const ollamaUrlInput = document.getElementById("ollama-url");
    const ollamaModelInput = document.getElementById("ollama-model");
    const ollamaStatusIcon = document.getElementById("ollama-status-icon");
    const ollamaStatusText = document.getElementById("ollama-status-text");
    let ollamaConnected = false;

    async function checkOllamaStatus() {
        const url = ollamaUrlInput.value.trim();
        const statusDiv = document.querySelector(".ollama-status");
        if (!statusDiv) return;
        
        try {
            const res = await fetch(`${url}/api/tags`, { method: 'GET' });
            if (res.ok) {
                ollamaConnected = true;
                statusDiv.className = "ollama-status connected";
                ollamaStatusIcon.className = "fa-solid fa-circle-check status-dot";
                ollamaStatusText.textContent = "Ollama On";
            } else {
                throw new Error();
            }
        } catch (e) {
            ollamaConnected = false;
            statusDiv.className = "ollama-status disconnected";
            ollamaStatusIcon.className = "fa-solid fa-circle-xmark status-dot";
            ollamaStatusText.textContent = "Ollama Off";
        }
    }

    // Check on startup and on input blur
    checkOllamaStatus();
    ollamaUrlInput.addEventListener("blur", checkOllamaStatus);
    ollamaModelInput.addEventListener("blur", checkOllamaStatus);

    // Chatbot send handler
    chatSendBtn.addEventListener("click", handleChatSubmit);
    chatInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") handleChatSubmit();
    });

    // Load initial registry JSON or setup mock baseline
    async function loadInitialData() {
        try {
            // Attempt to load registry summary
            const runRes = await fetch('last_run_web.json');
            if (runRes.ok) {
                const runData = await runRes.json();
                addRunToRegistry(runData);
            }
            
            const ragRes = await fetch('last_rag_web.json');
            if (ragRes.ok) {
                const ragData = await ragRes.json();
                ragLastRun.textContent = ragData.last_run_id;
                ragDocCount.textContent = ragData.document_count;
            }
            
            const indexRes = await fetch('eht_rag_index.json');
            if (indexRes.ok) {
                const indexData = await indexRes.json();
                indexedDocs = indexData.documents;
            }
        } catch (e) {
            console.log("No pre-existing JSON records found. Running with runtime mocks.", e);
        }
        
        // Add default mock runs if registry is empty
        if (modelRegistry.length === 0) {
            const baselineRun = {
                run_id: "run_1783159000",
                timestamp: "2026-07-04 07:00:00",
                status: "PRODUCTION",
                metrics: { fidelity_score: 0.9726, mse: 7.99e-8 },
                fit_result: { estimated_mass_10_9: 6.5, estimated_spin: 0.5 },
                promotion_log: "Initial approved benchmark model."
            };
            addRunToRegistry(baselineRun);
            
            // Build mock index document
            indexedDocs = [
                {
                    id: "run_1783159000_chunk_summary",
                    text: "O buraco negro supermassivo M87* foi observado e imageado na run run_1783159000. A massa estimada é de 6.5 bilhões de massas solares, com spin adimensional a = 0.50.",
                    metadata: { run_id: "run_1783159000", type: "summary" }
                },
                {
                    id: "run_1783159000_chunk_quality",
                    text: "A imagem reconstruída de M87* na run run_1783159000 atingiu um escore de fidelidade NCC de 97.26% e um Erro Quadrático Médio (MSE) de 7.990000e-08 contra a Relatividade Geral.",
                    metadata: { run_id: "run_1783159000", type: "quality" }
                },
                {
                    id: "run_1783159000_chunk_cal",
                    text: "O processamento na run run_1783159000 utilizou calibração baseada em Fase de Fechamento (Closure Phase) de 1.3mm, com parâmetros de regularização adicionais: alpha_tv = 0.05 e alpha_entropy = 0.005.",
                    metadata: { run_id: "run_1783159000", type: "calibration" }
                }
            ];
            
            ragLastRun.textContent = "run_1783159000";
            ragDocCount.textContent = indexedDocs.length;
        }
    }

    function addRunToRegistry(run) {
        modelRegistry.unshift(run);
        
        // Remove empty row
        if (emptyRunsRow) emptyRunsRow.remove();
        
        // Rebuild table rows
        runsTableBody.innerHTML = "";
        modelRegistry.forEach(r => {
            const tr = document.createElement("tr");
            
            const isApproved = r.status === "PRODUCTION";
            const gateClass = isApproved ? "approved" : "rejected";
            const gateText = isApproved ? "APROVADO" : "REJEITADO";
            
            tr.innerHTML = `
                <td>${r.run_id}</td>
                <td>${(r.metrics.fidelity_score * 100).toFixed(2)}%</td>
                <td>${r.fit_result.estimated_mass_10_9.toFixed(2)}B M☉</td>
                <td>${r.fit_result.estimated_spin.toFixed(2)}</td>
                <td><span class="badge-gate ${gateClass}">${gateText}</span></td>
                <td><span class="badge-deploy">${r.status}</span></td>
            `;
            runsTableBody.appendChild(tr);
        });
        
        // Update champion stats (highest approved NCC)
        const approved = modelRegistry.filter(r => r.status === "PRODUCTION");
        if (approved.length > 0) {
            const champ = approved.reduce((prev, current) => (prev.metrics.fidelity_score > current.metrics.fidelity_score) ? prev : current);
            championFidelity.textContent = `${(champ.metrics.fidelity_score * 100).toFixed(2)}%`;
            championMass.textContent = `${champ.fit_result.estimated_mass_10_9.toFixed(1)}B M☉`;
            championSpin.textContent = champ.fit_result.estimated_spin.toFixed(2);
        }
    }

    function resetDagNodesVisuals() {
        document.querySelectorAll(".dag-node").forEach(node => {
            node.className = "dag-node";
        });
    }

    function showDagWelcomeDetails(dag) {
        if (dag === "reconstruct") {
            detailsTitle.innerHTML = `<i class="fa-solid fa-rotate"></i> Pipeline de Reconstrução`;
            detailsStatusBadge.textContent = "PRONTO";
            detailsStatusBadge.className = "task-badge";
            detailsDesc.textContent = "DAG 01: Otimiza e reconstrói o anel de fótons relativístico do M87* por meio de regularizações RML fornecidas com gradientes analíticos.";
        } else if (dag === "drift") {
            detailsTitle.innerHTML = `<i class="fa-solid fa-satellite-dish"></i> Monitor de Drift Instrumental`;
            detailsStatusBadge.textContent = "PRONTO";
            detailsStatusBadge.className = "task-badge";
            detailsDesc.textContent = "DAG 02: Compara os perfis de visibilidade do scan atual contra a referência calibrada via PSI. Se PSI >= 0.20, aciona o retreino automático.";
        } else if (dag === "rag") {
            detailsTitle.innerHTML = `<i class="fa-solid fa-arrows-spin"></i> Indexador RAG / LLMOps`;
            detailsStatusBadge.textContent = "PRONTO";
            detailsStatusBadge.className = "task-badge";
            detailsDesc.textContent = "DAG 03: Extrai metadados físicos do modelo de Produção, vetoriza as informações com embeddings locais e atualiza o RAG Vector DB.";
        }
        plotArea.innerHTML = `
            <div class="visual-placeholder">
                <i class="fa-solid fa-rocket" style="font-size: 3rem; color: rgba(56,189,248,0.2); margin-bottom: 1rem;"></i>
                <span>Aguardando disparo da DAG Run.</span>
            </div>
        `;
    }

    // -------------------------------------------------------------
    // RUN PIPELINE 1: RECONSTRUCTION
    // -------------------------------------------------------------
    async function runReconstructionPipeline(isRetraining = false) {
        runningPipeline = "reconstruct";
        triggerBtn.disabled = true;
        triggerBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Executando...`;
        
        const steps = [
            { id: "node-r-ingest", name: "ingest_vlbi_data", desc: "Simulando coordenadas espaciais das baselines do telescópio na Terra e amostrando frequências de visibilidade.", duration: 3000, canvasType: "uv" },
            { id: "node-r-validate", name: "validate_schema", desc: "Validando dados VLBI. Checando colunas obrigatórias, presença de NaNs e coerência do fov_uas.", duration: 2000, canvasType: "validate" },
            { id: "node-r-calibrate", name: "calibrate_data", desc: "Autocalibração de fase. Modelando ruído térmico e aplicando estatística circular phasor complexa.", duration: 3000, canvasType: "calibrate" },
            { id: "node-r-reconstruct", name: "reconstruct_image", desc: "Minimizando a função objetivo combinada (Data Chi-sq + TV + Entropia) via L-BFGS-B com Gradiente Analítico.", duration: 5000, canvasType: "reconstruct" },
            { id: "node-r-evaluate", name: "evaluate_recon", desc: "Comparando a imagem resultante com a biblioteca teórica GRMHD. Estimando a massa e o spin de M87*.", duration: 3000, canvasType: "evaluate" },
            { id: "node-r-gate", name: "quality_gate", desc: "Validando limites físicos do Quality Gate MLOps: NCC >= 70%, massa e spin em faixas plausíveis.", duration: 2500, canvasType: "gate" }
        ];
        
        resetDagNodesVisuals();
        
        for (let i = 0; i < steps.length; i++) {
            const step = steps[i];
            const node = document.getElementById(step.id);
            node.classList.add("node-running");
            node.scrollIntoView({ behavior: "smooth", block: "nearest" });
            
            detailsTitle.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> ${step.name}`;
            detailsStatusBadge.textContent = "EXECUTANDO";
            detailsStatusBadge.className = "task-badge badge-running";
            detailsDesc.textContent = step.desc;
            
            // Build Canvas
            setupStepCanvas(step.canvasType);
            
            await sleep(step.duration);
            
            node.classList.remove("node-running");
            node.classList.add("node-success");
        }
        
        // Final task choice branch: register or reject
        const passGate = !isRetraining; // Let retraining pass, and first run pass.
        const resultStep = passGate 
            ? { id: "node-r-register", name: "register_model", desc: "Modelo aprovado no Quality Gate. Registrado no Model Registry e deploy em PRODUCTION concluído.", badge: "APROVADO", badgeClass: "badge-success" }
            : { id: "node-r-reject", name: "reject_model", desc: "Modelo falhou nos gates de qualidade. Salvo como FAILED_VALIDATION para depuração das regularizações.", badge: "REJEITADO", badgeClass: "badge-failed" };
            
        const resultNode = document.getElementById(resultStep.id);
        resultNode.classList.add(passGate ? "node-success" : "node-failed");
        
        detailsTitle.innerHTML = `<i class="fa-solid fa-shield-halved"></i> ${resultStep.name}`;
        detailsStatusBadge.textContent = resultStep.badge;
        detailsStatusBadge.className = `task-badge ${resultStep.badgeClass}`;
        detailsDesc.textContent = resultStep.desc;
        
        // Create run record
        const runId = "run_" + Math.floor(Date.now() / 1000).toString().substring(3);
        const newRun = {
            run_id: runId,
            timestamp: new Date().toISOString().replace("T", " ").substring(0, 19),
            status: passGate ? "PRODUCTION" : "FAILED_VALIDATION",
            metrics: {
                fidelity_score: passGate ? 0.9658 : 0.6212,
                mse: passGate ? 8.12e-8 : 9.54e-6
            },
            fit_result: {
                estimated_mass_10_9: passGate ? 6.50 : 4.20,
                estimated_spin: passGate ? 0.50 : -0.10
            },
            promotion_log: passGate ? "Champion model passed physical thresholds." : "Fidelity NCC too low or Mass violates GRMHD bounds."
        };
        
        // Add to Registry
        addRunToRegistry(newRun);
        
        // RAG refresh prompt
        if (passGate) {
            // Update temporary chunks database in RAM for chatbot queries
            updateLocalIndexDatabase(newRun);
        }
        
        // RAG index visual refresh values
        ragLastRun.textContent = runId;
        
        setupStepCanvas(passGate ? "register" : "reject");
        
        await sleep(3000);
        
        runningPipeline = null;
        triggerBtn.disabled = false;
        triggerBtn.innerHTML = `<i class="fa-solid fa-play"></i> Trigger DAG Run`;
    }

    // -------------------------------------------------------------
    // RUN PIPELINE 2: DRIFT MONITORING
    // -------------------------------------------------------------
    async function runDriftMonitoringPipeline() {
        runningPipeline = "drift";
        triggerBtn.disabled = true;
        triggerBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Executando...`;
        
        const steps = [
            { id: "node-d-ref", name: "extract_ref_data", desc: "Carregando coordenadas e amplitudes das visibilidades ideais (benchmarks de calibração).", duration: 2500, canvasType: "drift_ref" },
            { id: "node-d-scan", name: "extract_cur_scan", desc: "Amostrando visibilidades do scan diário e simulando ruídos atmosféricos adicionais configurados.", duration: 2500, canvasType: "drift_scan" },
            { id: "node-d-psi", name: "calculate_drift_psi", desc: "Calculando o Population Stability Index (PSI) entre as distribuições das amplitudes de rádio.", duration: 3000, canvasType: "drift_psi" },
            { id: "node-d-gate", name: "drift_gate", desc: "Checando se a dispersão e desvios nas fases (PSI) violam o limite estipulado de 0.20.", duration: 2000, canvasType: "drift_gate" }
        ];
        
        resetDagNodesVisuals();
        
        for (let i = 0; i < steps.length; i++) {
            const step = steps[i];
            const node = document.getElementById(step.id);
            node.classList.add("node-running");
            
            detailsTitle.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> ${step.name}`;
            detailsStatusBadge.textContent = "EXECUTANDO";
            detailsStatusBadge.className = "task-badge badge-running";
            detailsDesc.textContent = step.desc;
            
            setupStepCanvas(step.canvasType);
            await sleep(step.duration);
            node.classList.remove("node-running");
            node.classList.add("node-success");
        }
        
        // Decide drift trigger outcome
        const isAlert = driftSeverity === 3; // Severe drift triggers retraining
        const resultNode = document.getElementById("node-d-trigger");
        
        if (isAlert) {
            resultNode.classList.add("node-running");
            detailsTitle.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> trigger_retraining`;
            detailsStatusBadge.textContent = "ALERT (DRIFT DETECTED)";
            detailsStatusBadge.className = "task-badge badge-failed";
            detailsDesc.textContent = "Forte desvio de calibração instrumental detectado (PSI >= 0.20). O TriggerDagRunOperator disparou a execução automática do pipeline 01.";
            
            setupStepCanvas("drift_trigger_on");
            await sleep(3500);
            
            resultNode.classList.remove("node-running");
            resultNode.classList.add("node-success");
            
            // Auto switch to reconstruct DAG and run it
            await sleep(1000);
            const reconstructTab = document.querySelector('[data-dag="reconstruct"]');
            reconstructTab.click();
            
            // Run training automatically (retraining is successful/calibrated)
            runReconstructionPipeline(false);
        } else {
            resultNode.classList.add("node-success");
            detailsTitle.innerHTML = `<i class="fa-solid fa-circle-check"></i> drift_ok`;
            detailsStatusBadge.textContent = "OK (NORMAL)";
            detailsStatusBadge.className = "task-badge badge-success";
            detailsDesc.textContent = "Perfil do scan de rádio estável e coerente com a calibração de referência (PSI < 0.20). Retreino dispensado.";
            
            setupStepCanvas("drift_trigger_off");
            await sleep(3500);
        }
        
        runningPipeline = null;
        triggerBtn.disabled = false;
        triggerBtn.innerHTML = `<i class="fa-solid fa-play"></i> Trigger DAG Run`;
    }

    // -------------------------------------------------------------
    // RUN PIPELINE 3: RAG CATALOG REFRESH
    // -------------------------------------------------------------
    async function runRagRefreshPipeline() {
        runningPipeline = "rag";
        triggerBtn.disabled = true;
        triggerBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Executando...`;
        
        const steps = [
            { id: "node-g-extract", name: "extract_prod_model", desc: "Acessando o Model Registry e lendo os metadados físicos calculados da massa e do spin de M87*.", duration: 2500, canvasType: "rag_extract" },
            { id: "node-g-index", name: "update_vector_index", desc: "Fragmentando os dados em chunks de texto descritivos, gerando embeddings e inserindo no banco vetorial.", duration: 4000, canvasType: "rag_index" },
            { id: "node-g-fresh", name: "validate_freshness", desc: "Validando a consistência e o frescor das informações indexadas no eht_rag_index.json.", duration: 2500, canvasType: "rag_fresh" }
        ];
        
        resetDagNodesVisuals();
        
        for (let i = 0; i < steps.length; i++) {
            const step = steps[i];
            const node = document.getElementById(step.id);
            node.classList.add("node-running");
            
            detailsTitle.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> ${step.name}`;
            detailsStatusBadge.textContent = "EXECUTANDO";
            detailsStatusBadge.className = "task-badge badge-running";
            detailsDesc.textContent = step.desc;
            
            setupStepCanvas(step.canvasType);
            await sleep(step.duration);
            node.classList.remove("node-running");
            node.classList.add("node-success");
        }
        
        detailsTitle.innerHTML = `<i class="fa-solid fa-sparkles"></i> RAG Catalog Refreshed`;
        detailsStatusBadge.textContent = "CONCLUÍDO";
        detailsStatusBadge.className = "task-badge badge-success";
        detailsDesc.textContent = "Base de dados vetorial atualizada. O Chatbot agora possui acesso em tempo real aos novos dados físicos do buraco negro M87*.";
        
        ragDocCount.textContent = indexedDocs.length;
        
        setupStepCanvas("rag_completed");
        await sleep(3000);
        
        runningPipeline = null;
        triggerBtn.disabled = false;
        triggerBtn.innerHTML = `<i class="fa-solid fa-play"></i> Trigger DAG Run`;
    }

    // Update indexed documents list locally in memory
    function updateLocalIndexDatabase(run) {
        const runId = run.run_id;
        const mass = run.fit_result.estimated_mass_10_9;
        const spin = run.fit_result.estimated_spin;
        const ncc = run.metrics.fidelity_score;
        const mse = run.metrics.mse;
        
        // Recreate the three chunks
        const chunks = [
            {
                id: `${runId}_chunk_summary`,
                text: `O buraco negro supermassivo M87* foi observado e imageado na run ${runId}. A massa estimada é de ${mass.toFixed(2)} bilhões de massas solares, com spin adimensional a = ${spin.toFixed(2)}.`,
                metadata: { run_id: runId, type: "summary" }
            },
            {
                id: `${runId}_chunk_quality`,
                text: `A imagem reconstruída de M87* na run ${runId} atingiu um escore de fidelidade NCC de ${(ncc*100).toFixed(2)}% e um Erro Quadrático Médio (MSE) de ${mse.toExponential(6)} contra a Relatividade Geral.`,
                metadata: { run_id: runId, type: "quality" }
            },
            {
                id: `${runId}_chunk_cal`,
                text: `O processamento na run ${runId} utilizou calibração baseada em Fase de Fechamento (Closure Phase) de 1.3mm, com parâmetros de regularização adicionais: alpha_tv = 0.05 e alpha_entropy = 0.005.`,
                metadata: { run_id: runId, type: "calibration" }
            }
        ];
        
        // Remove existing chunks of this run if any
        indexedDocs = indexedDocs.filter(doc => doc.metadata.run_id !== runId);
        
        // Add new ones
        indexedDocs.push(...chunks);
    }

    // Helper functions
    function sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    // -------------------------------------------------------------
    // CANVAS RENDERERS (THE CORE MATHEMATICAL ANIMATIONS)
    // -------------------------------------------------------------
    function setupStepCanvas(type) {
        plotArea.innerHTML = "";
        const canvas = document.createElement("canvas");
        canvas.width = 460;
        canvas.height = 190;
        canvas.className = "mlops-canvas-small";
        plotArea.appendChild(canvas);
        
        const ctx = canvas.getContext("2d");
        let start = Date.now();
        let animationId;
        
        function renderLoop() {
            let elapsed = Date.now() - start;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Draw background grid
            ctx.strokeStyle = "rgba(56, 189, 248, 0.03)";
            ctx.lineWidth = 1;
            for (let x = 0; x < canvas.width; x += 20) {
                ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
            }
            for (let y = 0; y < canvas.height; y += 20) {
                ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
            }
            
            // Dispatch to specific animations
            if (type === "uv") {
                drawUVTracks(ctx, canvas, elapsed);
            } else if (type === "validate") {
                drawSchemaValidate(ctx, canvas, elapsed);
            } else if (type === "calibrate") {
                drawPhaseCalibration(ctx, canvas, elapsed);
            } else if (type === "reconstruct") {
                drawRMLOptimization(ctx, canvas, elapsed);
            } else if (type === "evaluate") {
                drawGRMHDFitting(ctx, canvas, elapsed);
            } else if (type === "gate") {
                drawQualityGateMetrics(ctx, canvas, elapsed);
            } else if (type === "register") {
                drawSuccessPromotion(ctx, canvas, elapsed);
            } else if (type === "reject") {
                drawRejectionStamp(ctx, canvas, elapsed);
            } else if (type === "drift_ref") {
                drawDriftReference(ctx, canvas, elapsed);
            } else if (type === "drift_scan") {
                drawDriftScan(ctx, canvas, elapsed);
            } else if (type === "drift_psi") {
                drawDriftPsiCalculation(ctx, canvas, elapsed);
            } else if (type === "drift_gate") {
                drawDriftGateDecision(ctx, canvas, elapsed);
            } else if (type === "drift_trigger_on") {
                drawDriftTriggerState(ctx, canvas, elapsed, true);
            } else if (type === "drift_trigger_off") {
                drawDriftTriggerState(ctx, canvas, elapsed, false);
            } else if (type === "rag_extract") {
                drawRagExtraction(ctx, canvas, elapsed);
            } else if (type === "rag_index") {
                drawRagIndexVectors(ctx, canvas, elapsed);
            } else if (type === "rag_fresh") {
                drawRagFreshnessCheck(ctx, canvas, elapsed);
            } else if (type === "rag_completed") {
                drawRagFinalState(ctx, canvas, elapsed);
            }
            
            animationId = requestAnimationFrame(renderLoop);
        }
        
        // Start animation loop
        renderLoop();
        
        // Store animation ID on canvas so it gets cleaned up if canvas is removed
        canvas.animationId = animationId;
    }

    // 1. Ingest: UV tracks
    function drawUVTracks(ctx, canvas, elapsed) {
        ctx.fillStyle = "rgba(56,189,248,0.05)";
        ctx.strokeStyle = "rgba(56,189,248,0.3)";
        ctx.beginPath();
        ctx.arc(canvas.width/2, canvas.height/2, 4, 0, Math.PI*2);
        ctx.fill(); ctx.stroke();
        
        // Draw baselines orbiting
        const baselines = [
            { rx: 60, ry: 40, rot: 0.1, color: "rgba(56,189,248,0.25)" },
            { rx: 110, ry: 50, rot: -0.3, color: "rgba(251,191,36,0.2)" },
            { rx: 140, ry: 75, rot: 0.5, color: "rgba(16,185,129,0.2)" }
        ];
        
        const speed = elapsed * 0.001;
        
        baselines.forEach(b => {
            ctx.save();
            ctx.translate(canvas.width/2, canvas.height/2);
            ctx.rotate(b.rot);
            ctx.strokeStyle = b.color;
            ctx.beginPath();
            ctx.ellipse(0, 0, b.rx, b.ry, 0, 0, Math.PI*2);
            ctx.stroke();
            
            // Draw baseline antennas paths dot
            const px = b.rx * Math.cos(speed);
            const py = b.ry * Math.sin(speed);
            ctx.fillStyle = "var(--accent-sky)";
            ctx.beginPath();
            ctx.arc(px, py, 3, 0, Math.PI*2);
            ctx.fill();
            
            // Draw grid dots
            for (let a = 0; a < Math.min(speed, Math.PI*2); a += 0.2) {
                const dx = b.rx * Math.cos(a);
                const dy = b.ry * Math.sin(a);
                ctx.fillStyle = "rgba(56, 189, 248, 0.4)";
                ctx.fillRect(dx, dy, 1, 1);
            }
            ctx.restore();
        });
        
        ctx.font = "9px Outfit";
        ctx.fillStyle = "rgba(255,255,255,0.7)";
        ctx.fillText("VLBI u-v frequencies coverage (8h Earth scan)", 10, 20);
        ctx.font = "bold 9px JetBrains Mono";
        ctx.fillStyle = "var(--accent-sky)";
        ctx.fillText("AMPLITUDES INGESTED: 1,500 POINTS", 10, 32);
    }

    // 2. Schema Validate checklist
    function drawSchemaValidate(ctx, canvas, elapsed) {
        const tests = [
            { name: "Verificando dimensões das baselines (15 caminhos)", pass: true, delay: 500 },
            { name: "Checando nulidades (NaN check)", pass: true, delay: 1000 },
            { name: "Validando declinação do M87* (12.39°)", pass: true, delay: 1500 }
        ];
        
        ctx.font = "bold 10px Outfit";
        ctx.fillStyle = "white";
        ctx.fillText("SCHEMA VALIDATION CHECKS (python_norte best practice)", 15, 25);
        
        tests.forEach((t, i) => {
            const py = 55 + i * 35;
            const active = elapsed >= t.delay;
            
            ctx.font = "9px Outfit";
            ctx.fillStyle = active ? "var(--text-main)" : "var(--text-muted)";
            ctx.fillText(t.name, 45, py);
            
            // Icon
            ctx.fillStyle = active ? "var(--accent-green)" : "rgba(148,163,184,0.1)";
            ctx.beginPath();
            ctx.arc(25, py - 3, 7, 0, Math.PI*2);
            ctx.fill();
            
            if (active) {
                ctx.font = "bold 9px Outfit";
                ctx.fillStyle = "#070913";
                ctx.fillText("✔", 22, py);
            }
        });
    }

    // 3. Calibrate phase: noise vs calibrated
    function drawPhaseCalibration(ctx, canvas, elapsed) {
        ctx.font = "9px Outfit";
        ctx.fillStyle = "white";
        ctx.fillText("Phasor Self-Calibration (EHT Paper III)", 15, 20);
        
        // Draw double plot: corrupted phase vs calibrated phase
        const width = 200;
        const height = 90;
        
        // Corrupted
        ctx.strokeStyle = "rgba(239, 68, 68, 0.4)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let x = 0; x < width; x++) {
            const noise = Math.sin(x*0.1) * 20 + Math.sin(x*0.5) * 10 + (Math.random() - 0.5) * 22;
            const px = 20 + x;
            const py = 90 + noise;
            if (x === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }
        ctx.stroke();
        
        // Calibrated
        ctx.strokeStyle = "var(--accent-green)";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        const progress = Math.min(elapsed / 3000, 1);
        for (let x = 0; x < width; x++) {
            const baseCurve = Math.sin(x*0.1) * 20;
            const noise = Math.sin(x*0.5) * 10 + (Math.random() - 0.5) * 22;
            // Morph from corrupted to clean
            const py = 90 + baseCurve + noise * (1 - progress);
            const px = 240 + x;
            if (x === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }
        ctx.stroke();
        
        ctx.font = "8px JetBrains Mono";
        ctx.fillStyle = "var(--accent-red)";
        ctx.fillText("Corrupted Phases (Atmosphere Noise)", 20, 160);
        ctx.fillStyle = "var(--accent-green)";
        ctx.fillText("Calibrated Phases (Phasor Mean)", 240, 160);
    }

    // 4. Reconstruct RML image morphing
    function drawRMLOptimization(ctx, canvas, elapsed) {
        const progress = Math.min(elapsed / 5000, 1);
        
        ctx.font = "9px Outfit";
        ctx.fillStyle = "white";
        ctx.fillText("RML Optimization Sky Image (L-BFGS-B Conjugate)", 15, 20);
        
        ctx.font = "bold 9px JetBrains Mono";
        ctx.fillStyle = "var(--accent-gold)";
        ctx.fillText(`ITERATION: ${Math.floor(progress * 60)} / 60`, 15, 32);
        
        // Render blurry image merging into a crisp ring
        const cx = canvas.width / 2;
        const cy = canvas.height / 2 + 10;
        
        // Draw back shadow
        ctx.save();
        ctx.shadowBlur = 10 + progress * 20;
        ctx.shadowColor = "rgba(249, 115, 22, 0.4)";
        
        // Blurry center
        ctx.fillStyle = "rgba(7, 9, 19, 1)";
        ctx.beginPath(); ctx.arc(cx, cy, 32, 0, Math.PI*2); ctx.fill();
        
        // Glow ring
        const grad = ctx.createRadialGradient(cx, cy, 10, cx, cy, 35);
        grad.addColorStop(0, 'rgba(7,9,19,1)');
        grad.addColorStop(0.5, `rgba(249,115,22, ${0.3 + progress * 0.7})`);
        grad.addColorStop(0.7, `rgba(236,72,153, ${0.15 + progress * 0.55})`);
        grad.addColorStop(1, 'rgba(7,9,19,0)');
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(cx, cy, 38, 0, Math.PI*2);
        ctx.fill();
        
        // Relativistic asymmetry crescent (more beaming on bottom left)
        if (progress > 0.3) {
            const CrescentGrad = ctx.createRadialGradient(cx - 8, cy + 8, 4, cx - 8, cy + 8, 26);
            CrescentGrad.addColorStop(0, `rgba(255,255,255, ${(progress - 0.3) * 1.2})`);
            CrescentGrad.addColorStop(0.4, `rgba(251,191,36, ${progress})`);
            CrescentGrad.addColorStop(1, 'rgba(251,191,36,0)');
            ctx.fillStyle = CrescentGrad;
            ctx.beginPath();
            ctx.arc(cx - 8, cy + 8, 26, 0, Math.PI*2);
            ctx.fill();
        }
        
        // Overlap shadow mask
        ctx.fillStyle = "#070913";
        ctx.beginPath();
        ctx.arc(cx, cy, 16 - progress * 1.5, 0, Math.PI*2);
        ctx.fill();
        
        ctx.restore();
    }

    // 5. Evaluate GRMHD fitting
    function drawGRMHDFitting(ctx, canvas, elapsed) {
        ctx.font = "9px Outfit";
        ctx.fillStyle = "white";
        ctx.fillText("GRMHD Theoretical Template Matching", 15, 20);
        
        // Show three library thumbnails, highlight the correct one
        const models = [
            { label: "M: 5.5B M☉ / a: 0.1", corr: 0.62 },
            { label: "M: 6.5B M☉ / a: 0.5 (M87*)", corr: 0.98, active: true },
            { label: "M: 7.5B M☉ / a: 0.9", corr: 0.74 }
        ];
        
        models.forEach((m, i) => {
            const rx = 35 + i * 150;
            const ry = 40;
            const rw = 120;
            const rh = 120;
            
            const isMatch = m.active && elapsed >= 1500;
            
            // Draw box
            ctx.fillStyle = isMatch ? "rgba(16,185,129,0.06)" : "rgba(15,23,42,0.4)";
            ctx.strokeStyle = isMatch ? "var(--accent-green)" : "rgba(56,189,248,0.15)";
            ctx.lineWidth = isMatch ? 2 : 1;
            ctx.beginPath();
            ctx.roundRect(rx, ry, rw, rh, 4);
            ctx.fill(); ctx.stroke();
            
            // Model name
            ctx.font = "8px Outfit";
            ctx.fillStyle = isMatch ? "var(--accent-green)" : "var(--text-muted)";
            ctx.fillText(m.label, rx + 10, ry + 16);
            
            // Correlation
            ctx.font = "bold 9px JetBrains Mono";
            ctx.fillStyle = isMatch ? "var(--accent-green)" : "var(--accent-sky)";
            ctx.fillText(`CORR: ${m.corr.toFixed(2)}`, rx + 10, ry + 32);
            
            // Drawing mini black hole models
            const mx = rx + 60;
            const my = ry + 80;
            
            ctx.fillStyle = "rgba(251,191,36,0.3)";
            ctx.beginPath(); ctx.arc(mx, my, 18, 0, Math.PI*2); ctx.fill();
            
            ctx.fillStyle = "black";
            ctx.beginPath(); ctx.arc(mx, my, 8, 0, Math.PI*2); ctx.fill();
            
            if (isMatch) {
                ctx.strokeStyle = "var(--accent-green)";
                ctx.beginPath(); ctx.arc(mx, my, 22, 0, Math.PI*2); ctx.stroke();
            }
        });
    }

    // 6. Quality Gate checklist
    function drawQualityGateMetrics(ctx, canvas, elapsed) {
        ctx.font = "bold 10px Outfit";
        ctx.fillStyle = "white";
        ctx.fillText("MLOPS QUALITY GATE STATS (Airflow validation)", 15, 25);
        
        const gates = [
            { name: "Image Fidelity Score (NCC >= 0.70)", val: "0.9658", pass: true, delay: 500 },
            { name: "Mass Range Boundaries (5.0 <= M <= 8.0)", val: "6.5B M☉", pass: true, delay: 1200 },
            { name: "Analytical Gradients Loss Converged", val: "Converged", pass: true, delay: 1800 }
        ];
        
        gates.forEach((g, i) => {
            const py = 60 + i * 35;
            const active = elapsed >= g.delay;
            
            if (active) {
                // Check
                ctx.fillStyle = "var(--accent-green)";
                ctx.font = "10px Outfit";
                ctx.fillText("✔", 20, py);
                
                // Details
                ctx.font = "9px Outfit";
                ctx.fillStyle = "var(--text-main)";
                ctx.fillText(g.name, 35, py);
                
                ctx.font = "bold 9px JetBrains Mono";
                ctx.fillStyle = "var(--accent-green)";
                ctx.fillText(g.val, canvas.width - 90, py);
            } else {
                ctx.font = "9px Outfit";
                ctx.fillStyle = "rgba(148,163,184,0.2)";
                ctx.fillText("Pendente...", 35, py);
            }
        });
    }

    // Register Approved Promotion
    function drawSuccessPromotion(ctx, canvas, elapsed) {
        // Star particle burst
        const alpha = Math.max(1 - elapsed / 2500, 0);
        ctx.fillStyle = `rgba(16, 185, 129, ${alpha * 0.15})`;
        ctx.fillRect(0,0,canvas.width,canvas.height);
        
        // Draw stamp "APPROVED & DEPLOYED"
        ctx.save();
        ctx.translate(canvas.width/2, canvas.height/2);
        ctx.rotate(-0.06);
        
        ctx.strokeStyle = "var(--accent-green)";
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.roundRect(-140, -35, 280, 70, 6);
        ctx.stroke();
        
        ctx.font = "bold 18px Outfit";
        ctx.fillStyle = "var(--accent-green)";
        ctx.textAlign = "center";
        ctx.fillText("MODEL PROMOTED", 0, -8);
        ctx.font = "bold 13px JetBrains Mono";
        ctx.fillText("STATUS: PRODUCTION", 0, 16);
        
        ctx.restore();
    }

    // Reject stamp
    function drawRejectionStamp(ctx, canvas, elapsed) {
        ctx.save();
        ctx.translate(canvas.width/2, canvas.height/2);
        ctx.rotate(0.04);
        
        ctx.strokeStyle = "var(--accent-red)";
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.roundRect(-140, -35, 280, 70, 6);
        ctx.stroke();
        
        ctx.font = "bold 18px Outfit";
        ctx.fillStyle = "var(--accent-red)";
        ctx.textAlign = "center";
        ctx.fillText("GATE CRITERIA FAILED", 0, -8);
        ctx.font = "bold 13px JetBrains Mono";
        ctx.fillText("STATUS: FAILED_VALIDATION", 0, 16);
        
        ctx.restore();
    }

    // Pipeline 2: Drift Reference
    function drawDriftReference(ctx, canvas, elapsed) {
        ctx.font = "9px Outfit";
        ctx.fillStyle = "white";
        ctx.fillText("Extracting Baseline Reference Calibration Data", 15, 20);
        
        // Draw baseline amplitude bar distribution
        ctx.strokeStyle = "rgba(56, 189, 248, 0.4)";
        ctx.lineWidth = 2;
        
        const pts = [20, 35, 60, 80, 75, 50, 42, 30, 22, 10];
        
        ctx.fillStyle = "rgba(56, 189, 248, 0.05)";
        ctx.fillRect(40, 50, 380, 90);
        
        pts.forEach((p, i) => {
            const bx = 60 + i * 36;
            const bh = p * Math.min(elapsed / 1500, 1);
            
            ctx.fillStyle = "var(--accent-sky)";
            ctx.beginPath();
            ctx.roundRect(bx, 140 - bh, 22, bh, 2);
            ctx.fill();
        });
        
        ctx.font = "8px JetBrains Mono";
        ctx.fillStyle = "var(--text-muted)";
        ctx.fillText("AMPLITUDES REFERENCE MEAN: 1.0", 40, 160);
    }

    // Pipeline 2: Drift Scan
    function drawDriftScan(ctx, canvas, elapsed) {
        ctx.font = "9px Outfit";
        ctx.fillStyle = "white";
        ctx.fillText("Extracting Current Daily Scan Visibilities", 15, 20);
        
        // Show current amplitudes (affected by noise based on severity)
        ctx.fillStyle = "rgba(251,191,36, 0.05)";
        ctx.fillRect(40, 50, 380, 90);
        
        const ptsReference = [20, 35, 60, 80, 75, 50, 42, 30, 22, 10];
        const multiplier = driftSeverity === 1 ? 0.95 : (driftSeverity === 2 ? 0.8 : 0.45);
        
        ptsReference.forEach((p, i) => {
            const bx = 60 + i * 36;
            // Add distortion depending on severity
            let offset = 0;
            if (driftSeverity === 2) offset = (Math.sin(i) * 5);
            if (driftSeverity === 3) offset = (Math.cos(i) * 15 + (Math.random() - 0.5) * 10);
            
            const bh = Math.max(p * multiplier + offset, 4) * Math.min(elapsed / 1500, 1);
            
            ctx.fillStyle = driftSeverity === 3 ? "var(--accent-red)" : "var(--accent-gold)";
            ctx.beginPath();
            ctx.roundRect(bx, 140 - bh, 22, bh, 2);
            ctx.fill();
        });
        
        ctx.font = "8px JetBrains Mono";
        ctx.fillStyle = "var(--text-muted)";
        ctx.fillText(`SIMULATED DRIFT LEVEL: ${driftSeverity === 3 ? "CRITICAL" : (driftSeverity === 2 ? "MODERATE" : "LOW")}`, 40, 160);
    }

    // Pipeline 2: Drift PSI Calculation
    function drawDriftPsiCalculation(ctx, canvas, elapsed) {
        ctx.font = "9px Outfit";
        ctx.fillStyle = "white";
        ctx.fillText("Drift PSI Calculation (Population Stability Index)", 15, 20);
        
        // Show mathematical overlap histogram
        ctx.fillStyle = "rgba(255,255,255,0.02)";
        ctx.fillRect(40, 40, 380, 90);
        
        const size = 15;
        const speed = elapsed * 0.005;
        
        // Draw reference curve (sky) and current curve (gold/red)
        ctx.strokeStyle = "var(--accent-sky)";
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let x = 0; x <= size; x++) {
            const rx = 50 + x * 25;
            const ry = 110 - Math.sin(x*0.25) * 45 - Math.cos(x*0.1) * 15;
            if (x === 0) ctx.moveTo(rx, ry); else ctx.lineTo(rx, ry);
        }
        ctx.stroke();
        
        ctx.strokeStyle = driftSeverity === 3 ? "var(--accent-red)" : "var(--accent-gold)";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        
        const driftShift = driftSeverity === 3 ? 12 : (driftSeverity === 2 ? 4 : 1);
        
        for (let x = 0; x <= size; x++) {
            const rx = 50 + x * 25;
            const ry = 110 - Math.sin((x - driftShift * Math.min(elapsed/2000, 1))*0.25) * 45 - Math.cos((x - driftShift * Math.min(elapsed/2000, 1))*0.1) * 15 + (driftSeverity === 3 ? (Math.sin(x)*8) : 0);
            if (x === 0) ctx.moveTo(rx, ry); else ctx.lineTo(rx, ry);
        }
        ctx.stroke();
        
        // Equation text
        const computedPSI = driftSeverity === 1 ? 0.042 : (driftSeverity === 2 ? 0.125 : 2.3715);
        ctx.font = "bold 10px JetBrains Mono";
        ctx.fillStyle = driftSeverity === 3 ? "var(--accent-red)" : (driftSeverity === 2 ? "var(--accent-gold)" : "var(--accent-green)");
        ctx.fillText(`PSI = SUM( (Actual_i - Ref_i) * ln(Actual_i / Ref_i) ) = ${computedPSI.toFixed(4)}`, 40, 155);
    }

    // Pipeline 2: Drift Gate Decision
    function drawDriftGateDecision(ctx, canvas, elapsed) {
        ctx.font = "9px Outfit";
        ctx.fillStyle = "white";
        ctx.fillText("Drift Gate Evaluation threshold: 0.20", 15, 20);
        
        const computedPSI = driftSeverity === 1 ? 0.042 : (driftSeverity === 2 ? 0.125 : 2.3715);
        const threshold = 0.20;
        const triggerRetrain = computedPSI >= threshold;
        
        ctx.font = "bold 13px JetBrains Mono";
        ctx.fillStyle = "white";
        ctx.fillText(`PSI = ${computedPSI.toFixed(4)}`, 160, 70);
        ctx.font = "9px Outfit";
        ctx.fillStyle = "var(--text-muted)";
        ctx.fillText(`Drift Threshold Limit: ${threshold.toFixed(2)}`, 160, 85);
        
        // Status box
        ctx.strokeStyle = triggerRetrain ? "var(--accent-red)" : "var(--accent-green)";
        ctx.fillStyle = triggerRetrain ? "rgba(239,68,68,0.08)" : "rgba(16,185,129,0.08)";
        ctx.beginPath();
        ctx.roundRect(160, 100, 200, 40, 4);
        ctx.fill(); ctx.stroke();
        
        ctx.font = "bold 11px Outfit";
        ctx.fillStyle = triggerRetrain ? "var(--accent-red)" : "var(--accent-green)";
        ctx.fillText(triggerRetrain ? "⚠️ DRIFT ALERT: RETRAIN REQUIRED" : "✔ STABLE: RETRAIN SKIPPED", 175, 125);
    }

    // Pipeline 2: Triggering state visual
    function drawDriftTriggerState(ctx, canvas, elapsed, isTriggered) {
        ctx.fillStyle = isTriggered ? "rgba(239,68,68,0.05)" : "rgba(16,185,129,0.05)";
        ctx.fillRect(0,0,canvas.width,canvas.height);
        
        ctx.save();
        ctx.translate(canvas.width/2, canvas.height/2);
        
        ctx.strokeStyle = isTriggered ? "var(--accent-red)" : "var(--accent-green)";
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.roundRect(-140, -35, 280, 70, 6);
        ctx.stroke();
        
        ctx.font = "bold 16px Outfit";
        ctx.fillStyle = isTriggered ? "var(--accent-red)" : "var(--accent-green)";
        ctx.textAlign = "center";
        
        if (isTriggered) {
            ctx.fillText("TRIGGERING DAG 01", 0, -8);
            ctx.font = "bold 11px JetBrains Mono";
            ctx.fillText("AUTOMATED RETRAINING ACTIVE", 0, 16);
            
            // Draw visual pulse glow
            const rad = 140 + Math.sin(elapsed * 0.01) * 15;
            ctx.strokeStyle = "rgba(239, 68, 68, 0.25)";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.roundRect(-rad, -35 - (rad-140)/2, rad*2, 70 + (rad-140), 6);
            ctx.stroke();
        } else {
            ctx.fillText("CALIBRATION STABLE", 0, -8);
            ctx.font = "bold 11px JetBrains Mono";
            ctx.fillText("SYSTEM MONITOR STATE: OK", 0, 16);
        }
        ctx.restore();
    }

    // Pipeline 3: RAG Extraction
    function drawRagExtraction(ctx, canvas, elapsed) {
        ctx.font = "9px Outfit";
        ctx.fillStyle = "white";
        ctx.fillText("Extracting Metadata from Champion Model", 15, 20);
        
        const items = [
            { key: "Run ID:", val: "run_1783161318" },
            { key: "Fidelity Score (NCC):", val: "97.26%" },
            { key: "Estimated Mass:", val: "6.5B M☉" },
            { key: "Estimated Spin (a):", val: "0.50" }
        ];
        
        items.forEach((item, i) => {
            const py = 60 + i * 25;
            ctx.font = "9px Outfit";
            ctx.fillStyle = "var(--text-muted)";
            ctx.fillText(item.key, 40, py);
            
            ctx.font = "bold 10px JetBrains Mono";
            ctx.fillStyle = "var(--accent-sky)";
            ctx.fillText(item.val, 190, py);
        });
    }

    // Pipeline 3: RAG vector chunks indexing
    function drawRagIndexVectors(ctx, canvas, elapsed) {
        ctx.font = "9px Outfit";
        ctx.fillStyle = "white";
        ctx.fillText("Generating Semantic Vector Embeddings", 15, 20);
        
        // Show text chunk on left, vector floats floating out on right
        const textChunk = "Chunk 1: O buraco negro supermassivo M87* tem uma massa de 6.5B massas solares...";
        ctx.font = "8px Outfit";
        ctx.fillStyle = "var(--text-muted)";
        ctx.fillText(textChunk.substring(0, 45) + "...", 15, 55);
        ctx.fillText("Vector size: 16 Dimensions", 15, 70);
        
        // Draw float vectors
        const progress = Math.min(elapsed / 4000, 1);
        
        ctx.strokeStyle = "rgba(56, 189, 248, 0.2)";
        ctx.beginPath();
        ctx.moveTo(15, 80); ctx.lineTo(250, 80);
        ctx.stroke();
        
        // Floats
        ctx.font = "9px JetBrains Mono";
        ctx.fillStyle = "var(--accent-gold)";
        
        const floatVals = [0.12, -0.42, 0.78, 0.05, -0.19, 0.33, 0.61, -0.02, 0.22, -0.51, 0.09, 0.17, 0.44, -0.38, 0.81, -0.11];
        
        floatVals.forEach((v, idx) => {
            const col = idx % 8;
            const row = Math.floor(idx / 8);
            const px = 200 + col * 32;
            const py = 100 + row * 24;
            
            ctx.save();
            ctx.fillStyle = `rgba(251,191,36, ${Math.min(elapsed / 1000 - idx*0.1, 1)})`;
            ctx.fillText(v.toFixed(2), px, py);
            ctx.restore();
        });
    }

    // Pipeline 3: Freshness check
    function drawRagFreshnessCheck(ctx, canvas, elapsed) {
        ctx.font = "bold 10px Outfit";
        ctx.fillStyle = "white";
        ctx.fillText("RAG INDEX FRESHNESS VALIDATION (LLMOps gate)", 15, 25);
        
        const checks = [
            { name: "Verificando consistência do index eht_rag_index.json", pass: true, delay: 500 },
            { name: "Verificando se o run_id atual está indexado", pass: true, delay: 1200 },
            { name: "Embedding vector units normalization validation", pass: true, delay: 1800 }
        ];
        
        checks.forEach((c, i) => {
            const py = 60 + i * 35;
            const active = elapsed >= c.delay;
            
            if (active) {
                ctx.fillStyle = "var(--accent-green)";
                ctx.font = "10px Outfit";
                ctx.fillText("✔", 20, py);
                
                ctx.font = "9px Outfit";
                ctx.fillStyle = "var(--text-main)";
                ctx.fillText(c.name, 35, py);
            } else {
                ctx.font = "9px Outfit";
                ctx.fillStyle = "rgba(148,163,184,0.2)";
                ctx.fillText("Aguardando checagem...", 35, py);
            }
        });
    }

    function drawRagFinalState(ctx, canvas, elapsed) {
        ctx.fillStyle = "rgba(56, 189, 248, 0.05)";
        ctx.fillRect(0,0,canvas.width,canvas.height);
        
        ctx.save();
        ctx.translate(canvas.width/2, canvas.height/2);
        
        ctx.strokeStyle = "var(--accent-sky)";
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.roundRect(-140, -35, 280, 70, 6);
        ctx.stroke();
        
        ctx.font = "bold 16px Outfit";
        ctx.fillStyle = "var(--accent-sky)";
        ctx.textAlign = "center";
        ctx.fillText("VECTOR DB UPDATED", 0, -8);
        ctx.font = "bold 11px JetBrains Mono";
        ctx.fillText("CHAT SYSTEM REFRESHED: OK", 0, 16);
        
        ctx.restore();
    }

    // -------------------------------------------------------------
    // RAG CHATBOT SEARCH & RETRIEVAL ENGINE
    // -------------------------------------------------------------
    // Simplified deterministic sin-based embedding hash matching Python's
    function generateJSDeterministicEmbedding(text) {
        const dim = 16;
        let vector = [];
        
        for (let i = 0; i < dim; i++) {
            let val = 0;
            for (let idx = 0; idx < text.length; idx++) {
                val += text.charCodeAt(idx) * (i + 1) * (idx + 1);
            }
            vector.push(Math.sin(val) * 0.5);
        }
        
        // Normalize unit length
        let norm = 0;
        for (let i = 0; i < dim; i++) {
            norm += vector[i] * vector[i];
        }
        norm = Math.sqrt(norm);
        if (norm > 0) {
            for (let i = 0; i < dim; i++) {
                vector[i] = vector[i] / norm;
            }
        }
        return vector;
    }

    async function handleChatSubmit() {
        const queryText = chatInput.value.trim();
        if (!queryText) return;
        
        // Display User Bubble
        appendChatMessage("user", queryText);
        chatInput.value = "";
        
        // Execute Cosine Similarity search over local indexedDocs
        const queryEmb = generateJSDeterministicEmbedding(queryText);
        
        let matches = [];
        indexedDocs.forEach(doc => {
            const docEmb = generateJSDeterministicEmbedding(doc.text);
            
            // Dot product since both are normalized to length 1
            let similarity = 0;
            for (let i = 0; i < 16; i++) {
                similarity += queryEmb[i] * docEmb[i];
            }
            
            // Apply keyword matching bonus to avoid orthogonal vector mismatches on simple synonyms
            const q = queryText.toLowerCase();
            let keywordBonus = 0;
            if (q.includes("massa") || q.includes("mass")) {
                if (doc.metadata.type === "summary") keywordBonus = 0.85;
            }
            if (q.includes("spin") || q.includes("rota") || q.includes("velocidade")) {
                if (doc.metadata.type === "summary" || doc.metadata.type === "calibration") keywordBonus = 0.85;
            }
            if (q.includes("fidelidade") || q.includes("ncc") || q.includes("imagem") || q.includes("qualidade")) {
                if (doc.metadata.type === "quality") keywordBonus = 0.85;
            }
            if (q.includes("calibra") || q.includes("fase") || q.includes("ruido") || q.includes("atmosfera")) {
                if (doc.metadata.type === "calibration") keywordBonus = 0.85;
            }
            
            const finalScore = Math.max(similarity, keywordBonus);
            matches.push({ doc: doc, score: finalScore });
        });
        
        // Sort by similarity score descending
        matches.sort((a, b) => b.score - a.score);
        
        // Top 2 context chunks
        const topMatches = matches.slice(0, 2);
        
        // Render retrieved context in UI panel
        retrievedText.innerHTML = "";
        topMatches.forEach((m, idx) => {
            const div = document.createElement("div");
            div.style.marginBottom = "0.4rem";
            div.innerHTML = `<span style="color:var(--accent-sky)">[Similarity: ${m.score.toFixed(3)}]</span> Run: ${m.doc.metadata.run_id} - "${m.doc.text}"`;
            retrievedText.appendChild(div);
        });
        
        const contextText = topMatches.map(m => m.doc.text).join("\n");
        const placeholderId = "msg-" + Date.now();
        appendChatMessage("system", `<i class="fa-solid fa-spinner fa-spin"></i> Pensando...`, placeholderId);
        
        if (ollamaConnected) {
            const url = ollamaUrlInput.value.trim();
            const model = ollamaModelInput.value.trim();
            
            try {
                const response = await fetch(`${url}/api/chat`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: model,
                        messages: [
                            {
                                role: "user",
                                content: `Instruções: Você é um assistente de IA astrofísico e especialista no Event Horizon Telescope. Responda à pergunta do usuário baseando-se estritamente no contexto fornecido (RAG). Seja técnico, conciso, direto ao ponto e responda em português.\n\nContexto:\n${contextText}\n\nPergunta: ${queryText}`
                            }
                        ],
                        stream: false
                    })
                });
                
                if (response.ok) {
                    const data = await response.json();
                    const aiReply = data.message.content;
                    updateChatMessage(placeholderId, aiReply);
                } else {
                    throw new Error("HTTP " + response.status);
                }
            } catch (err) {
                console.warn("Ollama failed, falling back to local template", err);
                const fallbackReply = generateFallbackLocalReply(queryText, topMatches);
                updateChatMessage(placeholderId, `*(Ollama Off - Resposta baseada em template)*<br><br>` + fallbackReply);
            }
        } else {
            // Local fallback directly
            await sleep(1000);
            const fallbackReply = generateFallbackLocalReply(queryText, topMatches);
            updateChatMessage(placeholderId, fallbackReply);
        }
    }

    function generateFallbackLocalReply(queryText, topMatches) {
        let answer = "";
        const bestMatch = topMatches[0];
        
        if (bestMatch && bestMatch.score > 0.15) {
            const rId = bestMatch.doc.metadata.run_id;
            const runInfo = modelRegistry.find(r => r.run_id === rId);
            
            if (runInfo) {
                const mass = runInfo.fit_result.estimated_mass_10_9;
                const spin = runInfo.fit_result.estimated_spin;
                const ncc = runInfo.metrics.fidelity_score;
                
                if (queryText.toLowerCase().includes("massa") || queryText.toLowerCase().includes("mass")) {
                    answer = `De acordo com a verificação semântica no catálogo EHT da **run ${rId}**, a massa física ajustada para M87* é de **${mass.toFixed(2)} bilhões de massas solares** ($6.5 \\times 10^9 M_\\odot$), demonstrando acoplamento preciso com as órbitas relativísticas teóricas de Schwarzschild.`;
                } else if (queryText.toLowerCase().includes("spin") || queryText.toLowerCase().includes("a")) {
                    answer = `As estimativas GRMHD na **run ${rId}** indicam um spin adimensional ajustado de **${spin.toFixed(2)}** (eixo de rotação voltado a ~135° Leste do Norte), o que corrobora o Doppler beaming relativístico do plasma no disco de acreção.`;
                } else if (queryText.toLowerCase().includes("fidelidade") || queryText.toLowerCase().includes("ncc") || queryText.toLowerCase().includes("imagem")) {
                    answer = `A reconstrução RML na **run ${rId}** obteve um escore de fidelidade de imagem (NCC) de **${(ncc*100).toFixed(2)}%** contra o modelo de relatividade geral de referência, superando o gate de qualidade física estabelecido de 70.00%.`;
                } else {
                    answer = `Recuperei metadados da **run ${rId}** no índice. Os resultados do imageamento do M87* indicam: Massa de **${mass.toFixed(1)}B M☉**, Spin de **${spin.toFixed(2)}**, e fidelidade estrutural NCC de **${(ncc*100).toFixed(2)}%** sob regularizações alpha_tv = 0.05 e alpha_entropy = 0.005.`;
                }
            } else {
                answer = `Encontrei informações relevantes indexadas na run **${rId}**: "${bestMatch.doc.text}"`;
            }
        } else {
            answer = "Não encontrei dados específicos sobre a sua pergunta no índice vetorial atual. Tente registrar um novo modelo aprovado na DAG 01 e rodar a indexação da DAG 03 para atualizar a base de conhecimento.";
        }
        return answer;
    }

    function appendChatMessage(sender, text, id = null) {
        const msgDiv = document.createElement("div");
        msgDiv.className = `chat-msg ${sender}`;
        if (id) msgDiv.id = id;
        
        const icon = sender === "user" 
            ? `<i class="fa-solid fa-user"></i>`
            : `<i class="fa-solid fa-robot"></i>`;
            
        msgDiv.innerHTML = `
            ${icon}
            <div class="msg-content">${text}</div>
        `;
        
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function updateChatMessage(id, text) {
        const msgDiv = document.getElementById(id);
        if (msgDiv) {
            const contentDiv = msgDiv.querySelector(".msg-content");
            if (contentDiv) {
                contentDiv.innerHTML = text.replace(/\n/g, "<br>");
            }
        }
    }

    // Set custom active tabs
    const subTabs = document.querySelectorAll(".tab-btn");
    const subContents = document.querySelectorAll(".tab-content");
    
    subTabs.forEach(tab => {
        tab.addEventListener("click", () => {
            subTabs.forEach(t => t.classList.remove("active"));
            subContents.forEach(c => c.classList.remove("active"));
            
            tab.classList.add("active");
            document.getElementById(`tab-${tab.dataset.tab}`).classList.add("active");
        });
    });
    
    // Show welcoming info
    showDagWelcomeDetails(activeDag);
});
