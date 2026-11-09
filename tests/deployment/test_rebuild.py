"""Regression checks for raw-row reconstruction and the native rebuild entry."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from deployment.commands import prepare_calibration
from deployment.rebuild import ROW, error_metrics, quality, validate_native_model, verify_normalized_rows, windows_from_rows
from shared.paths import ROOT


def rows(count=180):
    result=np.zeros(count,dtype=ROW)
    result['identity'][:,0]=1
    result['identity'][:,1]=np.arange(count)
    result['identity'][:,2]=np.arange(count)*3+1
    result['raw'][:,0]=10010+np.arange(count)
    result['raw'][:,2]=9990+np.arange(count)
    result['normalized']=np.arange(count*40,dtype='<f4').reshape(count,40)
    return result


class RebuildTests(unittest.TestCase):
    def test_chronological_window_and_integer_H50_labels(self):
        source=rows()
        values,labels=windows_from_rows(source,[99,100])
        self.assertEqual(values.shape,(2,40,100))
        np.testing.assert_array_equal(values[0],source['normalized'][:100].T)
        np.testing.assert_array_equal(values[1],source['normalized'][1:101].T)
        self.assertEqual(labels.tolist(),[2,2])

    def test_segment_boundary_is_rejected(self):
        source=rows()
        source['identity'][120:,0]=2
        with self.assertRaisesRegex(ValueError,'segment'):
            windows_from_rows(source,[99])

    def test_identity_gap_and_regression_are_rejected(self):
        for channel in (1,2):
            with self.subTest(channel=channel):
                source=rows()
                source['identity'][70,channel]=source['identity'][69,channel]
                with self.assertRaisesRegex(ValueError,'discontinuous'):
                    windows_from_rows(source,[99])

    def test_frozen_preprocessing_and_single_float_fault(self):
        source=rows(150)
        source['raw'][:,1::2]=100
        source['raw'][:,4::4]=10020
        source['raw'][:,6::4]=9980
        norm=json.loads((ROOT/'models/normalization.json').read_text('utf-8'))
        physical=source['raw'].astype('f8')
        midpoint=(physical[:,0]+physical[:,2])/2
        physical[:,::2]=(physical[:,::2]-midpoint[:,None])/midpoint[:,None]
        source['normalized']=((physical-np.array(norm['mean']))/np.array(norm['std'])).astype('<f4')
        verify_normalized_rows(source,norm)
        source['normalized'][20,4]=np.nextafter(source['normalized'][20,4],np.float32(np.inf))
        with self.assertRaisesRegex(ValueError,'preprocessing'):
            verify_normalized_rows(source,norm)

    def test_calibration_names_support_both_compile_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            inputs=directory/'sample.npy'
            np.save(inputs,np.zeros((2,40,100),dtype='<f4'))
            with patch('deployment.commands.work_path',side_effect=lambda path:directory/path):
                result=prepare_calibration(inputs)
            self.assertEqual(result['windows'],2)
            for name in ('inputs.npy','calibration.inputs.npy'):
                self.assertEqual((directory/'model_build/base/data'/name).read_bytes(),inputs.read_bytes())

    def test_reproduction_and_quality_metrics(self):
        logits=np.eye(3,dtype='<f4')
        self.assertTrue(error_metrics(logits,logits.copy())['bit_equal'])
        self.assertEqual(quality(logits,np.arange(3))['accuracy'],1.)
        self.assertEqual(quality(logits,np.arange(3))['macro_f1'],1.)
        with self.assertRaises(ValueError):
            error_metrics(logits,np.full_like(logits,np.nan))

    def test_native_bindings_and_hardware_compute_target(self):
        model=json.loads((ROOT/'models/npu/model.json').read_text('utf-8'))
        result=validate_native_model(model)
        self.assertEqual(result['layers'],30)
        self.assertEqual(result['input_bytes'],4800)
        self.assertEqual(result['output_id'],250)
        model['ops'][1]['compile_target']='@hostt'
        with self.assertRaisesRegex(ValueError,'non-ZG330'):
            validate_native_model(model)


if __name__=='__main__':
    unittest.main()
