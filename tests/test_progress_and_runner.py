"""Visible progress, original errors, and retry accounting are part of readiness."""
import runpy
import sys
import threading
import time
import subprocess
import torch
from atml_pa3.common import ROOT, amp_update
from atml_pa3.progress import stage


def test_stage_heartbeat_completion_and_failure(capsys):
    with stage('slow fixture', interval=0.01):
        time.sleep(0.035)
    err=capsys.readouterr().err
    assert 'START: slow fixture' in err and 'WORKING: slow fixture' in err and 'DONE: slow fixture' in err
    try:
        with stage('failed fixture', interval=0.01):
            raise ValueError('original error')
    except ValueError as e:
        assert str(e)=='original error'
    else:
        raise AssertionError('Stage suppressed the exception')
    assert 'FAILED: failed fixture' in capsys.readouterr().err


def test_colab_runner_streams_both_outputs_and_preserves_failure(tmp_path, capsys, monkeypatch):
    runner=runpy.run_path(str(ROOT/'scripts/colab_runner.py'))
    monkeypatch.setenv('MPLBACKEND','module://unavailable_colab_backend')
    command=[sys.executable,'-u','-c',"import sys,os; print(os.environ['MPLBACKEND']); print('stderr detail',file=sys.stderr); sys.exit(7)"]
    try:
        runner['run_logged'](command,tmp_path)
    except subprocess.CalledProcessError as e:
        assert e.returncode==7 and 'stderr detail' in e.output
    else:
        raise AssertionError('Runner suppressed failure')
    output=capsys.readouterr().out
    assert 'Agg' in output and 'stderr detail' in output
    log=next((tmp_path/'logs').glob('*.log')).read_text()
    assert 'stderr detail' in log and 'Agg' in log


def test_finite_gradients_with_overflowed_norm_are_retried():
    p=torch.nn.Parameter(torch.zeros(2))
    optimizer=torch.optim.AdamW([p],lr=0.1,weight_decay=0)
    scaler=torch.amp.GradScaler('cpu',init_scale=1)
    attempts=[];skips=[]
    def backward():
        scaler.scale(p.sum()).backward()
        attempts.append(1)
        p.grad.fill_(3e38 if len(attempts)==1 else 0.1)
        return {'sft':1.0}
    _,norm,retries=amp_update(optimizer,scaler,[p],backward,skips.append)
    assert retries==1 and len(skips)==1
    assert torch.isfinite(p).all() and optimizer.state[p]['step']==1
