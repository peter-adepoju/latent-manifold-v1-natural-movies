# V1 natural movie manifold project summary


I analysed Allen Brain Observatory V1 calcium-imaging responses during natural movie viewing.
The workflow produced validated trial tensors, real visual features extracted from the actual
Allen natural movie frames, population-level exploratory analyses, low-dimensional neural
representations, baseline decoding models, CEBRA/dRNN advanced models, geometry diagnostics,
event-triggered interpretability analyses, and manuscript-ready report assets.

The main stimulus-decoding result is that held-out-repeat V1 population activity predicts
continuous image-derived visual statistics, including RMS contrast, luminance variability,
orientation selectivity, spatial-frequency centroid, and total spectral power. I treat
dominant-orientation classification as secondary/exploratory because natural movie orientation
labels are imbalanced and temporally autocorrelated.

The latent-representation analyses show that PCA, UMAP, ISOMAP, and CEBRA produce distinct
low-dimensional descriptions of the same V1 population trajectory. Under block-wise movie-frame
cross-validation, latent embeddings showed limited real-feature predictivity, so I interpret
these embedding-feature regressions as descriptive rather than as strong predictive evidence.
The dRNN next-state model captured temporal structure in the learned CEBRA trajectory, but its
performance should be compared against the persistence baseline rather than interpreted alone.

The geometry and interpretability analyses showed heavy-tailed speed, curvature, and tangling
events in the CEBRA trajectory. These high-geometry events often coincide with visually
interpretable natural-movie frames, but geometry-stimulus correlations were modest; therefore,
I interpret them as descriptive alignment between latent neural geometry and movie structure,
not as causal evidence.


## Generated figures

- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\01_example_raw_dff_traces_experiment_500855614.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\01_example_raw_dff_traces_experiment_500964514.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\01_example_raw_dff_traces_experiment_501271265.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\01_selected_experiments_by_cre_depth_layer.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\03_cell_quality_distributions.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\03_population_activity_heatmap.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\03_population_activity_heatmap_experiment_500855614.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\04_confident_orientation_class_balance.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\04_dominant_orientation_class_balance.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\04_example_movie_frames_by_confident_orientation_class.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\04_example_movie_frames_by_orientation_class.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\04_example_movie_frames_by_orientation_class_spread.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\04_orientation_class_balance.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\04_population_activity_heatmap.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\04_population_activity_heatmap_robust.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\04_population_energy_vs_rms_contrast.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\04_population_response_energy.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\04_real_movie_frame_features.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\04_trial_variability_by_frame.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\04_zscored_population_energy_and_movie_features.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\05_isomap_latent_trajectory_frame_index.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\05_isomap_latent_trajectory_orientation_selectivity.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\05_isomap_latent_trajectory_rms_contrast.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\05_isomap_latent_trajectory_spatial_frequency.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\05_pca_cumulative_explained_variance.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\05_pca_latent_trajectory_frame_index.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\05_pca_latent_trajectory_orientation_selectivity.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\05_pca_latent_trajectory_rms_contrast.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\05_pca_latent_trajectory_spatial_frequency.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\05_representation_feature_predictivity_summary.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\05_umap_latent_trajectory_frame_index.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\05_umap_latent_trajectory_orientation_selectivity.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\05_umap_latent_trajectory_rms_contrast.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\05_umap_latent_trajectory_spatial_frequency.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\06_balanced_confident_orientation_decoding_comparison.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\06_best_non_null_real_feature_confusion_matrix.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\06_coarse_orientation_decoding_comparison.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\06_continuous_feature_decoding_r2.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\06_raw_cells_balanced_confident_orientation_repeat_heldout.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\07_cebra_latent_trajectory.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\07_cebra_latent_trajectory_luminance_std.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\07_cebra_latent_trajectory_orientation_selectivity.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\07_cebra_latent_trajectory_rms_contrast.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\07_cebra_latent_trajectory_spatial_frequency_centroid.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\07_cebra_vs_classical_embedding_predictivity.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\07_drnn_next_state_baseline_comparison.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\07_drnn_training_curve.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\08_best_classification_decoding_results.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\08_best_real_feature_decoding_by_embedding.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\08_cebra_vs_classical_embedding_summary.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\08_continuous_feature_decoding_r2.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\08_curvature_vs_decoding_accuracy.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\08_drnn_next_state_baseline_comparison.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\08_geometry_curvature_by_embedding.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\08_geometry_participation_ratio_by_embedding.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\09_cebra_curvature_timeseries.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\09_cebra_curvature_timeseries_clipped_p99.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\09_cebra_speed_timeseries.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\09_cebra_speed_timeseries_clipped_p99.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\09_cebra_tangling_timeseries.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\09_cebra_tangling_timeseries_clipped_p99.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\09_event_triggered_context_high_curvature_cebra.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\09_event_triggered_context_high_speed_cebra.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\09_event_triggered_context_high_tangling_cebra.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\09_geometry_stimulus_alignment_heatmap_cebra.png`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\figures\09_movie_frames_at_top_geometry_events_cebra.png`

## Generated tables

- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\00_workflow_plan.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\01_available_v1_natural_movie_cre_lines.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\01_selected_experiments_by_cre_depth_layer.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\01_selected_experiments_by_cre_line.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\02_validation_experiment_500855614.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\03_cell_quality_experiment_500855614.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\04_confident_orientation_class_distribution.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\04_dominant_orientation_bin_distribution.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\04_dominant_orientation_coarse_distribution.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\04_exploratory_summary.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\04_neural_population_energy_vs_real_stimulus_features.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\04_orientation_label_interpretation.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\05_frame_index_and_population_energy_baselines.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\05_geometry_summary.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\05_latent_embedding_real_feature_regression.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\05_latent_gain_over_temporal_baseline.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\05_latent_vs_time_feature_prediction.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\05_pca_explained_variance_session_500855614.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\05_representation_method_summary.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\06_balanced_confident_orientation_subset_balance.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\06_baseline_decoding_benchmark.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\06_best_non_null_confusion_matrix_summary.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\06_classification_target_balance.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\06_continuous_feature_decoding_benchmark.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\06_embedding_file_summary_session_500855614.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\06_primary_continuous_decoding_summary.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\07_cebra_metadata_session_500855614.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\07_cebra_representation_summary_session_500855614.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\07_cebra_vs_other_embeddings_feature_regression_session_500855614.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\07_drnn_next_state_baseline_comparison_session_500855614.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\07_drnn_next_state_metrics_session_500855614.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\07_drnn_training_history_session_500855614.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\08_best_classification_decoders.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\08_best_continuous_feature_decoders.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\08_best_decoding_accuracy_by_embedding.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\08_cebra_embedding_feature_benchmark_table.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\08_cebra_representation_summary_table.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\08_cebra_vs_classical_embedding_summary.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\08_continuous_feature_benchmark_table.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\08_drnn_baseline_comparison_summary.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\08_drnn_next_state_baseline_table.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\08_drnn_next_state_metrics_table.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\08_drnn_training_history_table.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\08_geometry_summary_by_embedding.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\08_integrated_benchmark_table.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\08_loaded_results_status.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\09_event_triggered_geometry_stimulus_context_cebra_session_500855614.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\09_event_triggered_geometry_stimulus_context_summary_cebra_session_500855614.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\09_extreme_geometry_events_cebra_session_500855614.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\09_extreme_geometry_events_with_stimulus_features_cebra_session_500855614.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\09_geometry_distribution_summary_cebra_session_500855614.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\09_geometry_peak_feature_enrichment_cebra_session_500855614.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\09_geometry_stimulus_alignment_cebra_session_500855614.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\09_geometry_timeseries_cebra_session_500855614.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\09_robust_geometry_summary_cebra_session_500855614.csv`
- `C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies\reports\tables\09_top_geometry_events_cebra_session_500855614.csv`