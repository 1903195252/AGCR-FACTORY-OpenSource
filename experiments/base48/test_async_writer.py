import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from .async_writer import BoundedWriter
from . import sample, augment
from . import test_sample as fixtures


class AsyncWriterTests(unittest.TestCase):
    def test_two_pending_bound_and_backpressure_before_next_decode(self):
        entered=threading.Event();release=threading.Event();available=threading.Event()
        def blocked():
            entered.set();release.wait(5)
        with BoundedWriter() as writer:
            writer.submit(blocked);self.assertTrue(entered.wait(2))
            writer.submit(lambda:None)
            self.assertEqual(len(writer.pending),2)
            waiter=threading.Thread(target=lambda:(writer.wait_for_slot(),available.set()))
            waiter.start()
            self.assertFalse(available.wait(.05))
            release.set();waiter.join(2)
            self.assertTrue(available.is_set())

    def test_failed_write_never_commits_and_resume_is_lossless(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(sample,'RGB_SHAPE',(16,2,2,3,3)),patch.object(sample,'DEPTH_SHAPE',(16,2,2,3)):
            root=Path(tmp);original=fixtures.SampleTests().arrays()
            with self.assertRaisesRegex(OSError,'disk failure'):
                with BoundedWriter() as writer:
                    futures={}
                    for cell in sample.CELLS:
                        if cell==(0,1):
                            def fail():raise OSError('disk failure')
                            futures[cell]=writer.submit(fail)
                        else:futures[cell]=writer.submit(sample.publish_cell_file,root,0,*cell,original)
                    writer.submit(sample.commit_draw_futures,root,0,101,futures,{}, {})
            self.assertFalse((root/'draw_00.json').exists())
            with BoundedWriter() as writer:
                futures={cell:writer.submit(sample.publish_cell_file,root,0,*cell,original) for cell in sample.CELLS}
                writer.submit(sample.commit_draw_futures,root,0,101,futures,{}, {})
            sample.validate_draw(root,0,101)
            for cell in sample.CELLS:
                with np.load(root/sample.file_name(*cell,0),allow_pickle=False) as loaded:
                    for key,value in original.items():
                        self.assertEqual(loaded[key].dtype,value.dtype)
                        np.testing.assert_array_equal(loaded[key],value)

    def test_stop_flushes_complete_draw_without_marking_corpus_complete(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(sample,'RGB_SHAPE',(16,2,2,3,3)),patch.object(sample,'DEPTH_SHAPE',(16,2,2,3)):
            root=Path(tmp)
            with self.assertRaises(sample.SamplingStopped):
                with BoundedWriter() as writer:
                    futures={cell:writer.submit(sample.publish_cell_file,root,0,*cell,fixtures.SampleTests().arrays()) for cell in sample.CELLS}
                    writer.submit(sample.commit_draw_futures,root,0,101,futures,{}, {})
                    raise sample.SamplingStopped('requested stop')
            sample.validate_draw(root,0,101)
            self.assertFalse((root/'completion.json').exists())

    def test_augmentation_validation_failure_has_no_commit_marker(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(augment,'validate_proposal',side_effect=ValueError('invalid proposal')):
            root=Path(tmp);augment.ensure_history(root/'history.npz',{'rgb':np.zeros(1)})
            row=dict(uid='episode:0',episode='episode')
            contract=dict(split_sha256='split',state_identity={'variant':'paired'})
            with self.assertRaisesRegex(ValueError,'invalid proposal'):
                with BoundedWriter() as writer:
                    writer.submit(augment.publish_proposal,root,row,0,101,{'rgb':np.zeros(1)},row,contract)
            self.assertFalse((root/'sample_00.json').exists())


if __name__=='__main__':unittest.main()
