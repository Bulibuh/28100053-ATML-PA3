"""Exercise the real pinned harness and stock filters on local synthetic data."""
import copy
from types import SimpleNamespace
import datasets
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from lm_eval import evaluator
from lm_eval.tasks import TaskManager
from lm_eval.api.task import ConfigurableTask
from atml_pa3.task1_eval import traced_hflm_class, pack_samples
from test_protocol import tiny


def local_lm():
    vocab={word:i for i,word in enumerate(['<pad>','<eos>','<unk>','Q','A',':','yes','no','one','two','2','3','4','The','answer','is','.'])}
    t=Tokenizer(models.WordLevel(vocab=vocab,unk_token='<unk>'))
    t.pre_tokenizer=pre_tokenizers.Whitespace()
    tok=PreTrainedTokenizerFast(tokenizer_object=t,pad_token='<pad>',eos_token='<eos>',unk_token='<unk>')
    tok.chat_template="{% for m in messages %}{{m['role']}}: {{m['content']}}\n{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
    return traced_hflm_class()(pretrained=tiny().eval(),tokenizer=tok,batch_size=2,max_length=256,device='cpu')


def test_stock_gsm_filters_sample_join_and_generation_trace(monkeypatch, tmp_path):
    data=datasets.Dataset.from_list([{'question':'one two?', 'answer':'two #### 2'},
                                     {'question':'two one?', 'answer':'one #### 3'}])
    monkeypatch.setattr(datasets,'load_dataset',lambda *args,**kwargs:datasets.DatasetDict(test=data))
    config=copy.deepcopy(TaskManager()._get_config('gsm8k_cot'))
    config['num_fewshot']=0
    task=ConfigurableTask(config=config)
    lm=local_lm()
    result=evaluator.simple_evaluate(model=lm,tasks=[task],num_fewshot=0,
        apply_chat_template=True,gen_kwargs={'max_gen_toks':4,'do_sample':False},
        log_samples=True,bootstrap_iters=0,random_seed=6304,numpy_random_seed=6304,
        torch_random_seed=6304,fewshot_random_seed=6304)
    from atml_pa3.common import write_json
    write_json(tmp_path / 'real_harness_result.json', result)
    samples=result['samples']['gsm8k_cot']
    assert len(samples)==4  # two documents, two stock answer filters
    rows=pack_samples('gsm8k_cot',samples,2,lm.rows)
    assert len(rows)==2
    assert set(rows[0]['filters'])=={'flexible-extract','strict-match'}
    assert all('truncated' in r and 'extracted_answer' in r for r in rows)
    assert 'exact_match,flexible-extract' in result['results']['gsm8k_cot']


def test_stock_boolq_completion_likelihood_without_chat(monkeypatch):
    data=datasets.Dataset.from_list([{'passage':'one two', 'question':'one?', 'label':0},
                                     {'passage':'two one', 'question':'two?', 'label':1}])
    monkeypatch.setattr(datasets,'load_dataset',lambda *args,**kwargs:datasets.DatasetDict(validation=data,train=data))
    task=ConfigurableTask(config=copy.deepcopy(TaskManager()._get_config('boolq')))
    lm=local_lm()
    def no_chat(*a,**kw): raise AssertionError('Likelihood benchmark used a chat template')
    lm.apply_chat_template=no_chat
    result=evaluator.simple_evaluate(model=lm,tasks=[task],num_fewshot=0,apply_chat_template=False,
                                     log_samples=True,bootstrap_iters=0)
    rows=pack_samples('boolq',result['samples']['boolq'],2,[])
    assert len(rows)==2 and 'acc,none' in result['results']['boolq']


def test_stock_arc_and_hellaswag_normalized_and_raw_metrics(monkeypatch):
    cases={
        'arc_easy':[{'question':'one two?', 'choices':{'label':['A','B','C','D'],'text':['one','two','yes','no']},'answerKey':'A'},
                    {'question':'two one?', 'choices':{'label':['A','B','C','D'],'text':['one','two','yes','no']},'answerKey':'B'}],
        'hellaswag':[{'ctx_a':'one', 'ctx_b':'two', 'activity_label':'one', 'endings':['one','two','yes','no'], 'label':'0'},
                     {'ctx_a':'two', 'ctx_b':'one', 'activity_label':'two', 'endings':['one','two','yes','no'], 'label':'1'}],
    }
    for name, records in cases.items():
        data=datasets.Dataset.from_list(records)
        monkeypatch.setattr(datasets,'load_dataset',lambda *args,**kwargs:datasets.DatasetDict(train=data,validation=data,test=data))
        task=ConfigurableTask(config=copy.deepcopy(TaskManager()._get_config(name)))
        lm=local_lm()
        lm.apply_chat_template=lambda *a,**kw: (_ for _ in ()).throw(AssertionError('Chat template was used'))
        result=evaluator.simple_evaluate(model=lm,tasks=[task],num_fewshot=0,apply_chat_template=False,
                                         log_samples=True,bootstrap_iters=0)
        assert len(pack_samples(name,result['samples'][name],2,[]))==2
        assert {'acc,none','acc_norm,none'} <= result['results'][name].keys()


def test_batched_qualitative_matches_single_prompts_and_restores_padding():
    from atml_pa3.task1_eval import generate_qualitative_batch
    lm=local_lm()
    # Force deterministic EOS on the first generated token, preserving padding tests.
    model=lm.model
    original=model.generate
    def generate(**kwargs):
        kwargs['max_new_tokens']=3
        kwargs['forced_eos_token_id']=lm.tokenizer.eos_token_id
        return original(**kwargs)
    model.generate=generate
    records=[{'prompt':'one','system':None},{'prompt':'one two one two one','system':None}]
    before=lm.tokenizer.padding_side
    batched=generate_qualitative_batch(model,lm.tokenizer,records)
    single=[generate_qualitative_batch(model,lm.tokenizer,[r])[0] for r in records]
    assert batched==single
    assert lm.tokenizer.padding_side==before
    assert all(not r['truncated'] for r in batched)
