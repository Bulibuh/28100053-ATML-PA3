"""Regression for the Colab dtype crash AFTER successful full GSM8K generation."""
import json
import numpy as np
import pytest
import torch
from atml_pa3.common import write_json, append_json
from atml_pa3.task1_eval import recover_gsm8k_scores


def test_harness_metadata_torch_dtype_and_device_are_json_safe(tmp_path):
    path=tmp_path/'harness.json'
    write_json(path,dict(config=dict(dtype=torch.float16,device=torch.device('cuda:0')),
                         model_dtype=torch.float32,numpy_dtype=np.dtype('float32'),
                         scalar=torch.tensor(0.5),array=np.array([1,2])))
    result=json.loads(path.read_text())
    assert result['config']==dict(dtype='torch.float16',device='cuda:0')
    assert result['model_dtype']=='torch.float32' and result['scalar']==0.5
    assert result['numpy_dtype']=='float32' and result['array']==[1,2]


def test_recover_completed_gsm8k_means_without_rewriting_samples(tmp_path):
    path=tmp_path/'gsm8k_cot_samples.jsonl'
    for ident,strict,flex in [(0,0,1),(1,1,1)]:
        append_json(path,dict(doc_id=ident,generation_trace={'token_ids':[1]},extracted_answer='1',
            filters={name:dict(metrics={'exact_match':value}) for name,value in
                     [('strict-match',strict),('flexible-extract',flex)]}))
    before=path.read_bytes()
    scores=recover_gsm8k_scores(path,expected=2)
    assert scores['exact_match,strict-match']==0.5 and scores['exact_match,flexible-extract']==1
    assert path.read_bytes()==before
    with pytest.raises(ValueError,match='exactly'):
        recover_gsm8k_scores(path,expected=1319)
    rows=path.read_text().splitlines()
    path.write_text(rows[0]+'\n'+rows[0]+'\n')
    with pytest.raises(ValueError,match='exactly'):
        recover_gsm8k_scores(path,expected=2)
