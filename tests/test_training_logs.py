"""No persistent per-update writes; checkpoint snapshots preserve training history."""
import json
import torch
from atml_pa3.training_logs import TrainingLogs
from atml_pa3.common import read_jsonl


def test_updates_and_overflows_stay_local_until_completion(tmp_path):
    drive=tmp_path/'drive'
    logs=TrainingLogs('lora4',local_root=tmp_path/'local')
    for update in range(1,51):
        logs.append('train',dict(update=update,sft_loss=1/update))
    logs.append('overflow',dict(update=40,attempt=1))
    assert not drive.exists()
    assert len(read_jsonl(logs.directory/'train.jsonl'))==50
    logs.publish(drive)
    assert len(read_jsonl(drive/'train.jsonl'))==50
    assert read_jsonl(drive/'overflow.jsonl')==[dict(update=40,attempt=1)]


def test_checkpoint_history_restores_journals_without_persistent_logs(tmp_path):
    original=TrainingLogs('full',local_root=tmp_path/'local')
    for update in range(1,313):
        original.append('train',dict(update=update,sft_loss=1))
    history=original.snapshot()
    checkpoint=tmp_path/'recovery.pt'
    torch.save(dict(log_history=history,update=312),checkpoint)
    original.append('train',dict(update=313,sft_loss=2))
    recovered=torch.load(checkpoint,weights_only=True)
    restored=TrainingLogs('full',initial=recovered['log_history'],local_root=tmp_path/'new_runtime')
    restored.append('train',dict(update=313,sft_loss=3))
    restored.publish(tmp_path/'drive')
    rows=read_jsonl(tmp_path/'drive/train.jsonl')
    assert [r['update'] for r in rows]==list(range(1,314))
    assert rows[-1]['sft_loss']==3 and len(history['train'])==312


def test_previous_drive_log_is_unchanged_until_final_sync_and_backed_up(tmp_path):
    drive=tmp_path/'drive';drive.mkdir()
    path=drive/'train.jsonl';path.write_text('{"update": 1, "sft_loss": 4}\n')
    initial=path.read_bytes()
    logs=TrainingLogs('full',initial={'train':read_jsonl(path)},local_root=tmp_path/'local')
    logs.append('train',dict(update=2,sft_loss=2))
    assert path.read_bytes()==initial
    logs.publish(drive)
    assert len(read_jsonl(path))==2
    backup=next(drive.glob('train_before_sync_*.jsonl'))
    assert backup.read_bytes()==initial


def test_training_command_syncs_after_export_and_resumes_checkpoint_history(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from atml_pa3 import task1
    from atml_pa3.common import read_jsonl as real_read, write_json
    artifacts=tmp_path/'drive'; monkeypatch.setattr(task1,'ARTIFACT_ROOT',artifacts)
    prepared=artifacts/'data/prepared/task1';prepared.mkdir(parents=True)
    write_json(prepared/'selection.json',dict(train_sha256='fixture'))
    monkeypatch.setattr(task1,'sha256',lambda p:'fixture')
    row=dict(input_ids=[1,2,3],labels=[-100,2,3])
    monkeypatch.setattr(task1,'read_jsonl',lambda p:[row]*40000 if str(p).endswith('prepared/task1/train.jsonl') else real_read(p))
    monkeypatch.setattr(task1,'config',lambda:dict(task1=dict(full_lr=2e-5,adapter_lr=2e-4,checkpoints=[1,2])))
    monkeypatch.setattr(task1,'model_spec',lambda:dict(model='fixture',revision='fixed'))
    monkeypatch.setattr(task1,'tokenizer',lambda:SimpleNamespace(pad_token_id=0))
    monkeypatch.setattr(task1,'require_cuda',lambda:None)
    for name in ['reset_peak_memory_stats','synchronize','empty_cache']:
        monkeypatch.setattr(torch.cuda,name,lambda:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    monkeypatch.setattr(task1,'device_batch',lambda b:b)
    from contextlib import nullcontext
    monkeypatch.setattr(torch,'autocast',lambda *a,**kw:nullcontext())
    original_scaler=torch.amp.GradScaler
    monkeypatch.setattr(torch.amp,'GradScaler',lambda *a,**kw:original_scaler('cpu'))
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();torch.manual_seed(1)
            self.weight=torch.nn.Parameter(torch.randn(4,4))
        def forward(self,input_ids,attention_mask):
            return SimpleNamespace(logits=self.weight[input_ids])
    monkeypatch.setattr(task1,'construct',lambda method:(Model(),{}))
    class Updates:
        def __init__(self,seq):self.seq=seq
        def __iter__(self):return (i for i in self.seq if i<2)
        def set_postfix(self,**kw):pass
    monkeypatch.setattr(task1,'tqdm',lambda seq,**kw:Updates(seq))
    original_logs=task1.TrainingLogs
    monkeypatch.setattr(task1,'TrainingLogs',lambda method,initial:original_logs(method,initial,local_root=tmp_path/'local'))
    exports=[]
    def export(model,path):
        # A completed command publishes logs only AFTER adapter export.
        exports.append(model.weight.detach().clone())
        if len(exports)==1:assert not (artifacts/'evidence/task1/lora4/train.jsonl').exists()
        torch.save({},path)
    monkeypatch.setattr(task1,'export_factors',export)
    task1.train(SimpleNamespace(method='lora4',resume=None))
    log=artifacts/'evidence/task1/lora4/train.jsonl'
    assert [r['update'] for r in real_read(log)]==[1,2]
    checkpoint=artifacts/'checkpoints/task1/lora4/update_1.pt'
    ck=torch.load(checkpoint,weights_only=False)
    assert len(ck['log_history']['train'])==1
    task1.train(SimpleNamespace(method='lora4',resume=str(checkpoint)))
    assert [r['update'] for r in real_read(log)]==[1,2]
    assert [r['update'] for r in real_read(log.with_name('checkpoints.jsonl'))]==[1,2]
    torch.testing.assert_close(exports[0],exports[1],rtol=0,atol=0)
