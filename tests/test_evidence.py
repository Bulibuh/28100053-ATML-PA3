"""Evidence tables must distinguish scores, percentage points, and missing runs."""
import json
import pandas as pd
import pytest
from atml_pa3 import evidence
from atml_pa3.common import write_json, append_json


def test_tables_from_measured_artifacts_and_missing_run_audit(tmp_path,monkeypatch):
    monkeypatch.setattr(evidence,'ARTIFACT_ROOT',tmp_path)
    root=tmp_path/'evidence/task1'
    for condition,score in [('baseline',0.25),('lora4',0.40)]:
        write_json(root/f'evaluation/{condition}/scores.json',{
            'gsm8k_cot':{'exact_match,flexible-extract':score,'exact_match_stderr,flexible-extract':0.01},
            'hellaswag':{'acc,none':score,'acc_norm,none':score},
            'boolq':{'acc,none':score},'arc_easy':{'acc,none':score,'acc_norm,none':score}})
    append_json(root/'lora4/train.jsonl',dict(update=1,sft_loss=2.0,lr=0.0001))
    write_json(root/'lora4/resources.json',dict(trainable_parameters=100,seconds=1,peak_gpu_bytes=200))
    append_json(root/'lora4/checkpoints.jsonl',dict(update=1250,bytes=300))
    evidence.build()
    table=pd.read_csv(root/'benchmark_table.csv').set_index('condition')
    assert table.loc['lora4','gsm8k_cot_percent']==40
    assert abs(table.loc['lora4','gsm8k_cot_delta_pp']-15)<1e-10
    assert pd.isna(table.loc['full','gsm8k_cot_percent'])
    comparison=pd.read_csv(root/'benchmark_comparison.csv')
    assert not comparison.metric.str.contains('stderr').any()
    assert (root/'learning_curves.png').exists()
    assert json.loads((root/'status.json').read_text())['status']=='incomplete'


@pytest.mark.parametrize("filled", [False, True])
def test_audit_annotation_boolean_serializes_to_console_and_file(tmp_path, monkeypatch, capsys, filled):
    monkeypatch.setattr(evidence, 'ARTIFACT_ROOT', tmp_path)
    root = tmp_path / 'evidence/task1'
    folder = root / 'qualitative'
    folder.mkdir(parents=True)
    cols = ['format_compliance', 'coherence_relevance', 'repetition', 'math_intrusion',
            'inappropriate_content', 'excerpt', 'human_notes']
    pd.DataFrame([{col: ('reviewed' if filled else '') for col in cols} for _ in range(42)]).to_csv(
        folder / 'annotations_to_complete.csv', index=False)
    evidence.audit()
    console = json.loads(capsys.readouterr().out)
    saved = json.loads((root / 'status.json').read_text())
    assert console == saved
    assert console['checks']['human_annotations_complete'] is filled
