"""Isolated real-process regressions. Never points at user's trained artifacts."""
import concurrent.futures
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
CLI, TRAIN, DRIVER = sys.argv[1:4]
sys.argv[1:] = []

def run(*args, ok=None):
    p = subprocess.run(list(map(str,args)), capture_output=True)
    if ok is not None and (p.returncode == 0) != ok:
        raise AssertionError((args,p.returncode,p.stdout,p.stderr))
    return p

class Publication(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="niyah-publication-")
        cls.root = Path(cls.temp.name)
        cls.corpus = cls.root/'corpus.txt'; cls.corpus.write_text('hello world hello world\n' * 4)
        cls.tok=cls.root/'tok';cls.shard=cls.root/'shard';cls.ck=cls.root/'base.ckpt';cls.cur=cls.root/'base.cur'
        run(CLI,'prepare','--corpus',cls.corpus,'--tokenizer-out',cls.tok,'--shard-out',cls.shard,'--target-vocab','258','--min-pair-frequency','1','--sequence-length','4',ok=True)
        run(TRAIN,'new','--tokenizer',cls.tok,'--shard',cls.shard,'--checkpoint-out',cls.ck,'--cursor-out',cls.cur,'--updates','1','--batch-size','1','--accumulation-steps','1','--model-seed','42','--data-seed','7','--context-length','4','--embedding-dim','8','--layers','1','--heads','2','--kv-heads','1','--ffn-hidden-dim','16','--rms-norm-eps','0.00001','--tie-word-embeddings','1','--learning-rate','0.001','--beta1','0.9','--beta2','0.999','--epsilon','0.00000001','--weight-decay','0','--max-grad-norm','1',ok=True)
    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()
    def setUp(self):
        self.d=self.root/self._testMethodName;self.d.mkdir()
    def assert_clean(self): self.assertEqual(list(self.d.rglob('*.niyah-stage-*')),[])
    def save(self,stem,fault='none',ok=True):
        ck=self.d/(stem+'.ckpt');cur=self.d/(stem+'.cur')
        p=run(DRIVER,'save',self.tok,self.ck,self.cur,ck,cur,fault,ok=ok)
        return ck,cur,p
    def resume(self,ck,cur,ok):
        run(TRAIN,'resume','--tokenizer',self.tok,'--shard',self.shard,'--checkpoint-in',ck,'--cursor-in',cur,'--checkpoint-out',self.d/'resume.ckpt','--cursor-out',self.d/'resume.cur','--updates','1','--batch-size','1','--accumulation-steps','1',ok=ok)
    def test_existing_and_alias(self):
        dest=self.d/'dest';dest.write_bytes(b'preserve');alias=self.d/'alias';os.link(dest,alias)
        for out in (dest,alias,self.d/'..'/self.d.name/'dest'):
            run(DRIVER,'publish',out,'overwrite','none',ok=False)
            self.assertEqual(dest.read_bytes(),b'preserve')
        self.assert_clean()
    def test_cli_early_failures(self):
        dest=self.d/'out';dest.write_bytes(b'preserve')
        for tok,corpus in ((self.d/'missing',self.corpus),(self.tok,self.d/'missing')):
            run(CLI,'shard','--tokenizer',tok,'--corpus',corpus,'--shard-out',dest,'--sequence-length','4',ok=False)
            self.assertEqual(dest.read_bytes(),b'preserve')
        run(CLI,'prepare','--corpus',self.d/'missing','--tokenizer-out',self.d/'new.tok','--shard-out',dest,'--target-vocab','258','--min-pair-frequency','1','--sequence-length','4',ok=False)
        self.assertEqual(dest.read_bytes(),b'preserve');self.assert_clean()
    def test_concurrent_no_replace(self):
        dest=self.d/'winner'
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
            jobs=list(pool.map(lambda i:run(DRIVER,'publish',dest,str(i),'none'),range(12)))
        wins=[i for i,p in enumerate(jobs) if p.returncode==0]
        self.assertEqual(len(wins),1);self.assertEqual(dest.read_text(),str(wins[0]));self.assert_clean()
    def test_partial_write_and_publish_failure(self):
        dest=self.d/'dest';dest.write_bytes(b'original')
        for fault in ('partial_write','publish'):
            p=run(DRIVER,'atomic',dest,'new-content',fault,ok=False)
            if fault == 'partial_write': self.assertIn(b'partial_staged_bytes=5',p.stdout)
            self.assertEqual(dest.read_bytes(),b'original');self.assert_clean()
        p=run(DRIVER,'atomic',self.d/'absent','partial','partial_write',ok=False)
        self.assertIn(b'partial_staged_bytes=3',p.stdout)
        self.assertFalse((self.d/'absent').exists());self.assert_clean()
    def test_invalid_destination(self):
        (self.d/'file').write_bytes(b'x')
        run(DRIVER,'publish',self.d/'file'/'child','data','none',ok=False)
        run(DRIVER,'publish',self.d/'missing'/'child','data','none',ok=False)
        run(DRIVER,'publish',self.d,'data','none',ok=False);self.assert_clean()
    def test_cleanup_error_visible(self):
        p=run(DRIVER,'publish',self.d/'dest','data','cleanup',ok=False)
        self.assertIn(b'publication_cleanup_failed=',p.stderr)
        self.assertTrue(list(self.d.glob('*.niyah-stage-*')))
        self.assertEqual((self.d/'dest').read_bytes(),b'data')
    def test_all_serializers_preserve_on_failure_and_legacy_bytes(self):
        for kind,source in (('tokenizer',self.tok),('shard',self.shard),('checkpoint',self.ck),('cursor',self.cur)):
            dest=self.d/kind;dest.write_bytes(b'original')
            for fault in ('partial_write','publish'):
                run(DRIVER,'serialize',kind,self.tok,self.shard,self.ck,self.cur,dest,fault,ok=False)
                self.assertEqual(dest.read_bytes(),b'original');self.assert_clean()
            run(DRIVER,'serialize',kind,self.tok,self.shard,self.ck,self.cur,dest,'none',ok=True)
            self.assertEqual(dest.read_bytes(),source.read_bytes());self.assert_clean()
    def test_segment_cli_rejection(self):
        ck=self.d/'segment.ckpt'
        run(DRIVER,'serialize','segment',self.tok,self.shard,self.ck,self.cur,ck,'none',ok=True)
        for mode,args in (('run',['--prompt','x','--max-new-tokens','1']),('eval',['--heldout',str(self.shard),'--format','shard'])):
            p=run(CLI,mode,'--tokenizer',self.tok,'--checkpoint',ck,*args,ok=False)
            self.assertIn(b'segment_conditioning_unsupported',p.stderr)
        p=run(TRAIN,'finetune','--tokenizer',self.tok,'--shard',self.shard,'--checkpoint-in',ck,'--checkpoint-out',self.d/'out.ckpt','--cursor-out',self.d/'out.cur','--updates','1','--batch-size','1','--accumulation-steps','1','--data-seed','7','--learning-rate','0.001','--beta1','0.9','--beta2','0.999','--epsilon','0.00000001','--weight-decay','0','--max-grad-norm','1',ok=False)
        self.assertIn(b'segment_conditioning_unsupported',p.stderr)
    def test_pair_faults(self):
        for fault in ('checkpoint_save','checkpoint_identity','cursor_bind','cursor_save','partial_write','after_checkpoint','after_cursor','before_complete'):
            ck,cur,_=self.save(fault,fault,False)
            self.assertTrue(Path(str(ck)+'.pending').exists())
            self.assertFalse(Path(str(ck)+'.complete').exists())
            run(DRIVER,'validate',ck,cur,ok=False)
            self.assert_clean()
    def test_pair_success_and_resume(self):
        ck,cur,_=self.save('pair')
        p=run(DRIVER,'validate',ck,cur,ok=True);self.assertIn(b'legacy=0',p.stdout)
        metadata=Path(str(ck)+'.complete').read_bytes()
        self.assertEqual(metadata,b'NIYAHPR1'+hashlib.sha256(ck.read_bytes()).digest()+hashlib.sha256(cur.read_bytes()).digest())
        self.resume(ck,cur,True);self.assert_clean()
    def test_pair_missing_metadata_rejected_by_consumer(self):
        ck,cur,_=self.save('pair');Path(str(ck)+'.complete').unlink()
        self.resume(ck,cur,False);self.assert_clean()
    def test_pair_malformed_and_stale(self):
        ck,cur,_=self.save('pair');marker=Path(str(ck)+'.complete');good=marker.read_bytes()
        for bad in (b'',good[:-1],b'garbage',good+b'x',good[:8]+b'x'*64):
            marker.write_bytes(bad);run(DRIVER,'validate',ck,cur,ok=False)
        marker.write_bytes(good)
        cur.write_bytes(self.cur.read_bytes()+b'x');run(DRIVER,'validate',ck,cur,ok=False)
        self.assert_clean()
    def test_pair_malformed_intent(self):
        ck,cur,_=self.save('pair');Path(str(ck)+'.pending').write_bytes(b'bad')
        self.resume(ck,cur,False)
    def test_legacy_compatibility(self):
        ck=self.d/'legacy.ckpt';cur=self.d/'legacy.cur';shutil.copyfile(self.ck,ck);shutil.copyfile(self.cur,cur)
        p=run(DRIVER,'validate',ck,cur,ok=True);self.assertIn(b'legacy=1',p.stdout)
        self.resume(ck,cur,True)
    def test_contenders_pair_and_existing_preserved(self):
        ck=self.d/'pair.ckpt';cur=self.d/'pair.cur'
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            jobs=list(pool.map(lambda _:run(DRIVER,'save',self.tok,self.ck,self.cur,ck,cur,'none'),range(4)))
        self.assertEqual(sum(p.returncode==0 for p in jobs),1)
        before=(ck.read_bytes(),cur.read_bytes());run(DRIVER,'save',self.tok,self.ck,self.cur,ck,cur,'none',ok=False)
        self.assertEqual(before,(ck.read_bytes(),cur.read_bytes()));self.assert_clean()
    @unittest.skipUnless(os.name=='nt','Windows sharing denial')
    def test_access_denied_does_not_overwrite(self):
        import ctypes
        from ctypes import wintypes
        k=ctypes.WinDLL('kernel32',use_last_error=True)
        k.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
        k.CreateFileW.restype=wintypes.HANDLE;k.CloseHandle.argtypes=[wintypes.HANDLE]
        dest=self.d/'locked';dest.write_bytes(b'locked-original')
        h=k.CreateFileW(str(dest),0x80000000,0,None,3,0,None)
        self.assertNotEqual(h,wintypes.HANDLE(-1).value)
        try:
            run(DRIVER,'atomic',dest,'bad','none',ok=False)
            run(CLI,'shard','--tokenizer',self.tok,'--corpus',self.corpus,'--shard-out',dest,'--sequence-length','4',ok=False)
        finally:k.CloseHandle(h)
        self.assertEqual(dest.read_bytes(),b'locked-original');self.assert_clean()

if __name__=='__main__':unittest.main(verbosity=2)
