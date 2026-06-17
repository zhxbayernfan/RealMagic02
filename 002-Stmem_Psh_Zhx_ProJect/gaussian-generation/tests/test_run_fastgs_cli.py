import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'run_fastgs.py'


def load_module():
    spec = importlib.util.spec_from_file_location('run_fastgs_for_test', SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class RunFastgsCliTest(unittest.TestCase):
    def test_direct_image_mode_accepts_images_output_and_ply(self):
        mod = load_module()
        old_argv = sys.argv[:]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            images = root / 'images'
            output = root / 'output'
            ply = root / 'result.ply'
            images.mkdir()
            sys.argv = [
                'run_fastgs.py',
                '--images', str(images),
                '--output', str(output),
                '--ply', str(ply),
                '--iterations', '12',
            ]
            try:
                args = mod.parse_args()
            finally:
                sys.argv = old_argv

            self.assertEqual(args.images, str(images))
            self.assertEqual(args.output, str(output))
            self.assertEqual(args.ply, str(ply))
            self.assertIsNone(args.config)
            self.assertEqual(args.iterations, 12)


if __name__ == '__main__':
    unittest.main()

class DirectImageModeBehaviorTest(unittest.TestCase):
    def test_main_in_direct_image_mode_builds_dataset_and_copies_ply(self):
        mod = load_module()
        old_argv = sys.argv[:]
        calls = []
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            images = root / 'images'
            output = root / 'output'
            ply = root / 'result.ply'
            images.mkdir()
            (images / '000001.jpg').write_bytes(b'fake')

            def fake_validate(source_dir):
                self.assertTrue((Path(source_dir) / 'rgb').exists())
                self.assertTrue((Path(source_dir) / 'intrinsics.yaml').exists())
                self.assertTrue((Path(source_dir) / 'poses').exists())
                calls.append(('validate', source_dir))

            def fake_colmap(source_dir, colmap_dir, voxel_size):
                sparse = Path(colmap_dir) / 'sparse' / '0'
                sparse.mkdir(parents=True)
                for name in ['cameras.bin', 'images.bin', 'points3D.bin']:
                    (sparse / name).write_bytes(b'x')
                calls.append(('colmap', source_dir, colmap_dir, voxel_size))

            def fake_training(colmap_dir, output_dir, params):
                final = Path(output_dir) / 'point_cloud' / f"iteration_{params['iterations']}" / 'point_cloud.ply'
                final.parent.mkdir(parents=True)
                final.write_bytes(b'ply-data')
                calls.append(('train', colmap_dir, output_dir, params['iterations']))

            mod.validate_source_dir = fake_validate
            mod.run_colmap_conversion = fake_colmap
            mod.run_fastgs_training = fake_training
            sys.argv = [
                'run_fastgs.py', '--images', str(images), '--output', str(output), '--ply', str(ply), '--iterations', '7'
            ]
            try:
                mod.main()
            finally:
                sys.argv = old_argv

            self.assertEqual(ply.read_bytes(), b'ply-data')
            self.assertTrue(any(c[0] == 'validate' for c in calls))
            self.assertTrue(any(c[0] == 'colmap' for c in calls))
            self.assertTrue(any(c[0] == 'train' for c in calls))
