# Numbers Audit

| value | section | location | json_path | json_key | tolerance | description |
|-------|---------|----------|-----------|----------|-----------|-------------|
| 0.5357 | 5.2 | Table 2 (tab:h1_accuracy) | paper/tests/fixtures/h1_panel.json | summary.infon_gnn_2hop | 0.01 | Infon+GNN accuracy at 2-hop depth |
| 0.4716 | 5.2 | Table 2 (tab:h1_accuracy) | paper/tests/fixtures/h1_panel.json | summary.infon_gnn_3hop | 0.01 | Infon+GNN accuracy at 3-hop depth |
| 0.5116 | 5.2 | Table 2 (tab:h1_accuracy) | paper/tests/fixtures/h1_panel.json | summary.infon_gnn_4hop | 0.01 | Infon+GNN accuracy at 4-hop depth |
| 0.5357 | 5.2 | Table 2 (tab:h1_accuracy) | paper/tests/fixtures/h1_panel.json | summary.infon_symbolic_2hop | 0.01 | Infon-symbolic accuracy at 2-hop depth |
| 0.4716 | 5.2 | Table 2 (tab:h1_accuracy) | paper/tests/fixtures/h1_panel.json | summary.infon_symbolic_3hop | 0.01 | Infon-symbolic accuracy at 3-hop depth |
| 0.5116 | 5.2 | Table 2 (tab:h1_accuracy) | paper/tests/fixtures/h1_panel.json | summary.infon_symbolic_4hop | 0.01 | Infon-symbolic accuracy at 4-hop depth |
| 0.5357 | 5.2 | Table 2 (tab:h1_accuracy) | paper/tests/fixtures/h1_panel.json | summary.flat_retrieval_2hop | 0.01 | Flat retrieval accuracy at 2-hop depth |
| 0.4716 | 5.2 | Table 2 (tab:h1_accuracy) | paper/tests/fixtures/h1_panel.json | summary.flat_retrieval_3hop | 0.01 | Flat retrieval accuracy at 3-hop depth |
| 0.5116 | 5.2 | Table 2 (tab:h1_accuracy) | paper/tests/fixtures/h1_panel.json | summary.flat_retrieval_4hop | 0.01 | Flat retrieval accuracy at 4-hop depth |
| 0.3571 | 5.2 | Table 2 (tab:h1_accuracy) | paper/tests/fixtures/h1_panel.json | summary.symbolic_floor_2hop | 0.01 | Symbolic floor accuracy at 2-hop depth |
| 0.4236 | 5.2 | Table 2 (tab:h1_accuracy) | paper/tests/fixtures/h1_panel.json | summary.symbolic_floor_3hop | 0.01 | Symbolic floor accuracy at 3-hop depth |
| 0.3643 | 5.2 | Table 2 (tab:h1_accuracy) | paper/tests/fixtures/h1_panel.json | summary.symbolic_floor_4hop | 0.01 | Symbolic floor accuracy at 4-hop depth |
| 0.9281 | 5.3 | Table 3 (tab:h2_theta) | paper/tests/fixtures/h2_panel.json | summary.infon_gnn_theta_nei | 0.01 | Infon+GNN mean theta on NEI claims |
| 0.9319 | 5.3 | Table 3 (tab:h2_theta) | paper/tests/fixtures/h2_panel.json | summary.infon_gnn_theta_supports | 0.01 | Infon+GNN mean theta on SUPPORTS claims |
| 0.9204 | 5.3 | Table 3 (tab:h2_theta) | paper/tests/fixtures/h2_panel.json | summary.infon_gnn_theta_refutes | 0.01 | Infon+GNN mean theta on REFUTES claims |
| 0.9281 | 5.3 | Table 3 (tab:h2_theta) | paper/tests/fixtures/h2_panel.json | summary.infon_symbolic_theta_nei | 0.01 | Infon-symbolic mean theta on NEI claims |
| 0.9319 | 5.3 | Table 3 (tab:h2_theta) | paper/tests/fixtures/h2_panel.json | summary.infon_symbolic_theta_supports | 0.01 | Infon-symbolic mean theta on SUPPORTS claims |
| 0.9204 | 5.3 | Table 3 (tab:h2_theta) | paper/tests/fixtures/h2_panel.json | summary.infon_symbolic_theta_refutes | 0.01 | Infon-symbolic mean theta on REFUTES claims |
| 0.15 | 5.3 | Table 3 (tab:h2_theta) | paper/tests/fixtures/h2_panel.json | summary.llm_zeroshot_theta_nei | 0.01 | LLM zero-shot mean theta on NEI claims |
| 0.05 | 5.3 | Table 3 (tab:h2_theta) | paper/tests/fixtures/h2_panel.json | summary.llm_zeroshot_theta_supports | 0.01 | LLM zero-shot mean theta on SUPPORTS claims |
| 0.05 | 5.3 | Table 3 (tab:h2_theta) | paper/tests/fixtures/h2_panel.json | summary.llm_zeroshot_theta_refutes | 0.01 | LLM zero-shot mean theta on REFUTES claims |
| 0.0137 | 5.3 | prose | paper/tests/fixtures/h2_panel.json | summary.infon_gnn_spearman_rho | 0.01 | Infon+GNN Spearman rho(theta, NEI_indicator) |
| 0.0137 | 5.3 | prose | paper/tests/fixtures/h2_panel.json | summary.infon_symbolic_spearman_rho | 0.01 | Infon-symbolic Spearman rho(theta, NEI_indicator) |
| 0.3417 | 5.3 | prose | paper/tests/fixtures/h2_panel.json | summary.llm_zeroshot_spearman_rho | 0.01 | LLM zero-shot Spearman rho(theta, NEI_indicator) |
| 0.0 | 5.3 | prose | paper/tests/fixtures/h2_panel.json | summary.infon_gnn_fpr_nei | 0.01 | Infon+GNN false-positive endorsement rate on NEI |
| 0.0 | 5.3 | prose | paper/tests/fixtures/h2_panel.json | summary.infon_symbolic_fpr_nei | 0.01 | Infon-symbolic false-positive endorsement rate on NEI |
| 0.1507 | 5.3 | prose | paper/tests/fixtures/h2_panel.json | summary.llm_zeroshot_fpr_nei | 0.01 | LLM zero-shot false-positive endorsement rate on NEI |
| 0.2102 | 5.4 | Table 4 (tab:h3_calibration) | paper/tests/fixtures/h3_panel.json | summary.infon_gnn_ece | 0.01 | Infon+GNN ECE on SciFact |
| 0.2666 | 5.4 | Table 4 (tab:h3_calibration) | paper/tests/fixtures/h3_panel.json | summary.infon_gnn_brier | 0.01 | Infon+GNN Brier score on SciFact |
| 0.5764 | 5.4 | Table 4 (tab:h3_calibration) | paper/tests/fixtures/h3_panel.json | summary.infon_gnn_aurc | 0.01 | Infon+GNN AURC on SciFact |
| 0.2102 | 5.4 | Table 4 (tab:h3_calibration) | paper/tests/fixtures/h3_panel.json | summary.infon_symbolic_ece | 0.01 | Infon-symbolic ECE on SciFact |
| 0.2666 | 5.4 | Table 4 (tab:h3_calibration) | paper/tests/fixtures/h3_panel.json | summary.infon_symbolic_brier | 0.01 | Infon-symbolic Brier score on SciFact |
| 0.5764 | 5.4 | Table 4 (tab:h3_calibration) | paper/tests/fixtures/h3_panel.json | summary.infon_symbolic_aurc | 0.01 | Infon-symbolic AURC on SciFact |
| 0.3927 | 5.4 | Table 4 (tab:h3_calibration) | paper/tests/fixtures/h3_panel.json | summary.nli_classifier_ece | 0.01 | NLI classifier ECE on SciFact |
| 0.3961 | 5.4 | Table 4 (tab:h3_calibration) | paper/tests/fixtures/h3_panel.json | summary.nli_classifier_brier | 0.01 | NLI classifier Brier score on SciFact |
| 0.3951 | 5.4 | Table 4 (tab:h3_calibration) | paper/tests/fixtures/h3_panel.json | summary.nli_classifier_aurc | 0.01 | NLI classifier AURC on SciFact |
| 0.4716 | 5.4 | prose | paper/tests/fixtures/aggregate_panel.json | summary.infon_gnn_accuracy | 0.01 | Infon+GNN aggregate accuracy |
| 0.5764 | 5.4 | prose | paper/tests/fixtures/aggregate_panel.json | summary.infon_gnn_aurc | 0.01 | Infon+GNN aggregate AURC |
| 0.4716 | 5.4 | prose | paper/tests/fixtures/aggregate_panel.json | summary.infon_symbolic_accuracy | 0.01 | Infon-symbolic aggregate accuracy |
| 0.5764 | 5.4 | prose | paper/tests/fixtures/aggregate_panel.json | summary.infon_symbolic_aurc | 0.01 | Infon-symbolic aggregate AURC |
| 0.0 | 5.4 | prose | paper/tests/fixtures/aggregate_panel.json | summary.llm_zeroshot_accuracy | 0.01 | LLM zero-shot aggregate accuracy |
| 0.0 | 5.4 | prose | paper/tests/fixtures/aggregate_panel.json | summary.llm_zeroshot_aurc | 0.01 | LLM zero-shot aggregate AURC |
| 0.5033 | 5.4 | prose | paper/tests/fixtures/aggregate_panel.json | summary.nli_classifier_accuracy | 0.01 | NLI classifier aggregate accuracy |
| 0.3951 | 5.4 | prose | paper/tests/fixtures/aggregate_panel.json | summary.nli_classifier_aurc | 0.01 | NLI classifier aggregate AURC |
