"""Selection order/eligibility and reusable qualitative artifacts without downloads."""
import json
from types import SimpleNamespace
import datasets
import huggingface_hub
from atml_pa3 import task1, task1_eval, taskdefs
from atml_pa3.common import ROOT, read_jsonl, sha256, write_json
from test_protocol import TokenizerStub


def test_prepare_first_40000_eligible_in_source_order(tmp_path,monkeypatch):
    monkeypatch.setattr(task1,'ARTIFACT_ROOT',tmp_path)
    monkeypatch.setattr(taskdefs,'dataset_spec',lambda **kw:dict(revisions={'openai/gsm8k':'g','meta-math/MetaMathQA':'m'}))
    monkeypatch.setattr(huggingface_hub,'model_info',lambda *a,**kw:SimpleNamespace(sha='immutable-model-sha'))
    class Tok(TokenizerStub):
        def encode(self,s,**kw):return list(range(2050)) if s=='long' else [20,21,22]
        def apply_chat_template(self,messages,**kw):
            if kw.get('tokenize') is False:return 'non-thinking '+messages[0]['content']
            return super().apply_chat_template(messages,**kw)
    monkeypatch.setattr(task1,'tokenizer',lambda:Tok())
    class Rows(list): _fingerprint='synthetic-test-fixture'
    source=Rows([{'query':'Forbidden!', 'response':'a'},
                 {'query':'different', 'original_question':'FORBIDDEN', 'response':'a'},
                 {'query':' ', 'response':'a'}, {'query':'q','response':'long'}])
    source.extend(dict(query=f'question {i}',response='supplied answer') for i in range(40001))
    monkeypatch.setattr(datasets,'load_dataset',lambda name,*a,**kw:Rows([{'question':'forbidden'}]) if name=='openai/gsm8k' else source)
    class ProbeTask:
        def __init__(self,n):self.n=n
        def has_test_docs(self):return True
        def test_docs(self):return [{'text':str(i)} for i in range(self.n)]
        def doc_to_text(self,doc):return doc['text']
    monkeypatch.setattr(taskdefs,'load_tasks',lambda names:{'gsm8k_cot':ProbeTask(1319),'hellaswag':ProbeTask(10042)})
    task1.prepare(SimpleNamespace())
    root=tmp_path/'data/prepared/task1'
    rows=read_jsonl(root/'train.jsonl')
    assert len(rows)==40000
    assert [r['source_index'] for r in rows]==list(range(4,40004))
    selection=json.loads((root/'selection.json').read_text())
    assert selection['rejected']=={'empty':1,'contamination':2,'too_long':1}
    probes=json.loads((root/'probe_ids.json').read_text())
    assert len(probes)==64 and sum(r['dataset']=='gsm8k_cot' for r in probes)==32
    # Repeated prepare checks the saved artifacts instead of resampling.
    before=sha256(root/'probe_ids.json')
    task1.prepare(SimpleNamespace())
    assert sha256(root/'probe_ids.json')==before


def test_qualitative_reuses_all_conditions_and_preserves_human_notes(tmp_path,monkeypatch):
    monkeypatch.setattr(task1_eval,'ARTIFACT_ROOT',tmp_path)
    monkeypatch.setattr(task1_eval,'require_cuda',lambda:None)
    monkeypatch.setattr(task1_eval,'tokenizer',lambda:None)
    spec={'model':'fixture','revision':'fixed'}
    monkeypatch.setattr(task1_eval,'model_spec',lambda:spec)
    def should_not_generate(*a,**kw):raise AssertionError('Regenerated a completed condition')
    monkeypatch.setattr(task1_eval,'merged_model',should_not_generate)
    prompts=ROOT/'configs/first_plot_questions.yaml'
    records,questions=task1_eval.load_qualitative_prompts(prompts)
    out=tmp_path/'evidence/task1/qualitative'
    for condition in task1_eval.CONDITIONS:
        rows=[dict(**r,condition=condition,response='fixture response',truncated=False,
                   annotation_required=r['question_id'] in [questions[0],questions[-1]]) for r in records]
        write_json(out/f'{condition}.json',dict(model=spec,prompt_sha256=sha256(prompts),rows=rows))
    args=SimpleNamespace(prompts=str(prompts),condition=None)
    task1_eval.qualitative(args)
    rows=json.loads((out/'responses.json').read_text())
    assert len(rows)==168 and sum(r['annotation_required'] for r in rows)==42
    annotation=out/'annotations_to_complete.csv'
    annotation.write_text(annotation.read_text()+'\n# human note to preserve\n')
    before=annotation.read_bytes()
    task1_eval.qualitative(args)
    assert annotation.read_bytes()==before
