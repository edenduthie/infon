# Numbers Audit

| value | section | location | json_path | json_key | tolerance | description |
|-------|---------|----------|-----------|----------|-----------|-------------|
| 74.0 | 5.2 | Table 2 (tab:h1_accuracy) | paper/tests/fixtures/h1_panel.json | summary.cognition_gnn_2hop | 0.01 | Cognition+GNN accuracy at 2-hop depth |
| 70.0 | 5.2 | Table 2 (tab:h1_accuracy) | paper/tests/fixtures/h1_panel.json | summary.cognition_gnn_3hop | 0.01 | Cognition+GNN accuracy at 3-hop depth |
| 66.0 | 5.2 | Table 2 (tab:h1_accuracy) | paper/tests/fixtures/h1_panel.json | summary.cognition_gnn_4hop | 0.01 | Cognition+GNN accuracy at 4-hop depth |
| 72.0 | 5.2 | Table 2 (tab:h1_accuracy) | paper/tests/fixtures/h1_panel.json | summary.cognition_symbolic_2hop | 0.01 | Cognition-symbolic accuracy at 2-hop depth |
| 65.0 | 5.2 | Table 2 (tab:h1_accuracy) | paper/tests/fixtures/h1_panel.json | summary.cognition_symbolic_3hop | 0.01 | Cognition-symbolic accuracy at 3-hop depth |
| 58.0 | 5.2 | Table 2 (tab:h1_accuracy) | paper/tests/fixtures/h1_panel.json | summary.cognition_symbolic_4hop | 0.01 | Cognition-symbolic accuracy at 4-hop depth |
| 68.0 | 5.2 | Table 2 (tab:h1_accuracy) | paper/tests/fixtures/h1_panel.json | summary.flat_retrieval_2hop | 0.01 | Flat retrieval accuracy at 2-hop depth |
| 55.0 | 5.2 | Table 2 (tab:h1_accuracy) | paper/tests/fixtures/h1_panel.json | summary.flat_retrieval_3hop | 0.01 | Flat retrieval accuracy at 3-hop depth |
| 42.0 | 5.2 | Table 2 (tab:h1_accuracy) | paper/tests/fixtures/h1_panel.json | summary.flat_retrieval_4hop | 0.01 | Flat retrieval accuracy at 4-hop depth |
| 45.0 | 5.2 | Table 2 (tab:h1_accuracy) | paper/tests/fixtures/h1_panel.json | summary.symbolic_floor_2hop | 0.01 | Symbolic floor accuracy at 2-hop depth |
| 40.0 | 5.2 | Table 2 (tab:h1_accuracy) | paper/tests/fixtures/h1_panel.json | summary.symbolic_floor_3hop | 0.01 | Symbolic floor accuracy at 3-hop depth |
| 35.0 | 5.2 | Table 2 (tab:h1_accuracy) | paper/tests/fixtures/h1_panel.json | summary.symbolic_floor_4hop | 0.01 | Symbolic floor accuracy at 4-hop depth |
| 0.90 | 5.3 | Table 3 (tab:h2_theta) | paper/tests/fixtures/h2_panel.json | summary.cognition_gnn_theta_nei | 0.01 | Cognition+GNN mean theta on NEI claims |
| 0.13 | 5.3 | Table 3 (tab:h2_theta) | paper/tests/fixtures/h2_panel.json | summary.cognition_gnn_theta_supports | 0.01 | Cognition+GNN mean theta on SUPPORTS claims |
| 0.17 | 5.3 | Table 3 (tab:h2_theta) | paper/tests/fixtures/h2_panel.json | summary.cognition_gnn_theta_refutes | 0.01 | Cognition+GNN mean theta on REFUTES claims |
| 0.88 | 5.3 | Table 3 (tab:h2_theta) | paper/tests/fixtures/h2_panel.json | summary.cognition_symbolic_theta_nei | 0.01 | Cognition-symbolic mean theta on NEI claims |
| 0.15 | 5.3 | Table 3 (tab:h2_theta) | paper/tests/fixtures/h2_panel.json | summary.cognition_symbolic_theta_supports | 0.01 | Cognition-symbolic mean theta on SUPPORTS claims |
| 0.18 | 5.3 | Table 3 (tab:h2_theta) | paper/tests/fixtures/h2_panel.json | summary.cognition_symbolic_theta_refutes | 0.01 | Cognition-symbolic mean theta on REFUTES claims |
| 0.60 | 5.3 | Table 3 (tab:h2_theta) | paper/tests/fixtures/h2_panel.json | summary.llm_zeroshot_theta_nei | 0.01 | LLM zero-shot mean theta on NEI claims |
| 0.35 | 5.3 | Table 3 (tab:h2_theta) | paper/tests/fixtures/h2_panel.json | summary.llm_zeroshot_theta_supports | 0.01 | LLM zero-shot mean theta on SUPPORTS claims |
| 0.36 | 5.3 | Table 3 (tab:h2_theta) | paper/tests/fixtures/h2_panel.json | summary.llm_zeroshot_theta_refutes | 0.01 | LLM zero-shot mean theta on REFUTES claims |
| 0.95 | 5.3 | prose | paper/tests/fixtures/h2_panel.json | summary.cognition_gnn_spearman_rho | 0.01 | Cognition+GNN Spearman rho(theta, NEI_indicator) |
| 0.93 | 5.3 | prose | paper/tests/fixtures/h2_panel.json | summary.cognition_symbolic_spearman_rho | 0.01 | Cognition-symbolic Spearman rho(theta, NEI_indicator) |
| 0.52 | 5.3 | prose | paper/tests/fixtures/h2_panel.json | summary.llm_zeroshot_spearman_rho | 0.01 | LLM zero-shot Spearman rho(theta, NEI_indicator) |
| 0.03 | 5.3 | prose | paper/tests/fixtures/h2_panel.json | summary.cognition_gnn_fpr_nei | 0.01 | Cognition+GNN false-positive endorsement rate on NEI |
| 0.04 | 5.3 | prose | paper/tests/fixtures/h2_panel.json | summary.cognition_symbolic_fpr_nei | 0.01 | Cognition-symbolic false-positive endorsement rate on NEI |
| 0.24 | 5.3 | prose | paper/tests/fixtures/h2_panel.json | summary.llm_zeroshot_fpr_nei | 0.01 | LLM zero-shot false-positive endorsement rate on NEI |
| 0.05 | 5.4 | Table 4 (tab:h3_calibration) | paper/tests/fixtures/h3_panel.json | summary.cognition_gnn_ece | 0.01 | Cognition+GNN ECE on SciFact |
| 0.12 | 5.4 | Table 4 (tab:h3_calibration) | paper/tests/fixtures/h3_panel.json | summary.cognition_gnn_brier | 0.01 | Cognition+GNN Brier score on SciFact |
| 0.15 | 5.4 | Table 4 (tab:h3_calibration) | paper/tests/fixtures/h3_panel.json | summary.cognition_gnn_aurc | 0.01 | Cognition+GNN AURC on SciFact |
| 0.08 | 5.4 | Table 4 (tab:h3_calibration) | paper/tests/fixtures/h3_panel.json | summary.cognition_symbolic_ece | 0.01 | Cognition-symbolic ECE on SciFact |
| 0.16 | 5.4 | Table 4 (tab:h3_calibration) | paper/tests/fixtures/h3_panel.json | summary.cognition_symbolic_brier | 0.01 | Cognition-symbolic Brier score on SciFact |
| 0.20 | 5.4 | Table 4 (tab:h3_calibration) | paper/tests/fixtures/h3_panel.json | summary.cognition_symbolic_aurc | 0.01 | Cognition-symbolic AURC on SciFact |
| 0.12 | 5.4 | Table 4 (tab:h3_calibration) | paper/tests/fixtures/h3_panel.json | summary.nli_classifier_ece | 0.01 | NLI classifier ECE on SciFact |
| 0.22 | 5.4 | Table 4 (tab:h3_calibration) | paper/tests/fixtures/h3_panel.json | summary.nli_classifier_brier | 0.01 | NLI classifier Brier score on SciFact |
| 0.30 | 5.4 | Table 4 (tab:h3_calibration) | paper/tests/fixtures/h3_panel.json | summary.nli_classifier_aurc | 0.01 | NLI classifier AURC on SciFact |
| 70.0 | 5.4 | prose | paper/tests/fixtures/aggregate_panel.json | summary.cognition_gnn_accuracy | 0.01 | Cognition+GNN aggregate accuracy |
| 0.15 | 5.4 | prose | paper/tests/fixtures/aggregate_panel.json | summary.cognition_gnn_aurc | 0.01 | Cognition+GNN aggregate AURC |
| 65.0 | 5.4 | prose | paper/tests/fixtures/aggregate_panel.json | summary.cognition_symbolic_accuracy | 0.01 | Cognition-symbolic aggregate accuracy |
| 0.20 | 5.4 | prose | paper/tests/fixtures/aggregate_panel.json | summary.cognition_symbolic_aurc | 0.01 | Cognition-symbolic aggregate AURC |
| 62.0 | 5.4 | prose | paper/tests/fixtures/aggregate_panel.json | summary.llm_zeroshot_accuracy | 0.01 | LLM zero-shot aggregate accuracy |
| 0.28 | 5.4 | prose | paper/tests/fixtures/aggregate_panel.json | summary.llm_zeroshot_aurc | 0.01 | LLM zero-shot aggregate AURC |
| 60.0 | 5.4 | prose | paper/tests/fixtures/aggregate_panel.json | summary.nli_classifier_accuracy | 0.01 | NLI classifier aggregate accuracy |
| 0.30 | 5.4 | prose | paper/tests/fixtures/aggregate_panel.json | summary.nli_classifier_aurc | 0.01 | NLI classifier aggregate AURC |
