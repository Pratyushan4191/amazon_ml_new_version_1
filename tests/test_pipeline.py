import json

import numpy as np
import pandas as pd

from src import predict
from src.features import REQUIRED_COLUMNS, load_source_records


def test_load_source_records_supports_train_and_test_prefixes(tmp_path):
	for split in ("train", "test"):
		data_dir = tmp_path / split
		data_dir.mkdir()
		for source_number in (1, 2, 3):
			pd.DataFrame(
				[{
					"entity_id": f"S{source_number}-001",
					"business_name": "Example Co",
					"business_address": "1 Main St",
					"country": "US",
				}],
				columns=REQUIRED_COLUMNS,
			).to_csv(
				data_dir / f"{split}_source{source_number}.tsv",
				sep="\t",
				index=False,
			)

		sources = load_source_records(data_dir, split=split)
		assert sources["source1"].iloc[0]["entity_id"] == "S1-001"


def test_inference_outputs_s1_entities_without_candidates(tmp_path, monkeypatch):
	class StubBooster:
		def __init__(self, model_file):
			pass

		# Accepts **kwargs because the real lightgbm.Booster.predict does
		# (run_inference passes num_threads=...); a stub that does not will
		# fail on the signature rather than on the behaviour under test.
		def predict(self, features, **kwargs):
			return np.array([0.9])

	monkeypatch.setattr(predict.lgb, "Booster", StubBooster)
	feature_row = {
		"source1_entity_id": "S1-001",
		"candidate_entity_id": "S2-001",
		**{column: 0.0 for column in predict.FEATURE_COLS},
	}
	features_path = tmp_path / "features.parquet"
	pd.DataFrame([feature_row]).to_parquet(features_path, index=False)
	threshold_path = tmp_path / "threshold.json"
	threshold_path.write_text(json.dumps({"t_star": 0.5}), encoding="utf-8")
	output_path = tmp_path / "matching_results.tsv"

	predict.run_inference(
		features_path,
		tmp_path / "model.lgb",
		threshold_path,
		output_path,
		all_s1_ids=["S1-001", "S1-002"],
	)

	result = pd.read_csv(output_path, sep="\t", keep_default_na=False)
	assert result["source1_entity_id"].tolist() == ["S1-001", "S1-002"]
	assert result.loc[1, "matched_entity_ids"] == ""
