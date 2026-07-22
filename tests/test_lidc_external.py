import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lidc_external import (
    crop_patch_3x64,
    extract_external_patches,
    parse_lidc_xml,
    parse_xml_collection,
    resolve_slice_index,
)


class LIDCXmlTest(unittest.TestCase):
    def test_namespace_xml_centroid(self):
        xml = """<?xml version='1.0'?>
        <LidcReadMessage xmlns='http://www.nih.gov'>
          <ResponseHeader><SeriesInstanceUid>1.2.3</SeriesInstanceUid></ResponseHeader>
          <readingSession><unblindedReadNodule><noduleID>N1</noduleID>
            <roi><imageZposition>-100.0</imageZposition>
              <edgeMap><xCoord>10</xCoord><yCoord>20</yCoord></edgeMap>
              <edgeMap><xCoord>14</xCoord><yCoord>24</yCoord></edgeMap>
            </roi>
          </unblindedReadNodule></readingSession>
        </LidcReadMessage>"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotation.xml"
            path.write_text(xml, encoding="utf-8")
            uid, annotations = parse_lidc_xml(path)
        self.assertEqual(uid, "1.2.3")
        self.assertEqual(len(annotations), 1)
        self.assertAlmostEqual(annotations[0]["pixel_x"], 12.0)
        self.assertAlmostEqual(annotations[0]["pixel_y"], 22.0)
        self.assertAlmostEqual(annotations[0]["world_z"], -100.0)

    def test_sop_uid_is_kept_and_exclusion_roi_is_ignored(self):
        xml = """<?xml version='1.0'?>
        <LidcReadMessage xmlns='http://www.nih.gov'>
          <ResponseHeader><SeriesInstanceUid>1.2.4</SeriesInstanceUid></ResponseHeader>
          <readingSession><unblindedReadNodule><noduleID>N2</noduleID>
            <roi><imageZposition>-10</imageZposition><imageSOP_UID>SOP-A</imageSOP_UID>
              <inclusion>TRUE</inclusion>
              <edgeMap><xCoord>10</xCoord><yCoord>20</yCoord></edgeMap>
              <edgeMap><xCoord>12</xCoord><yCoord>22</yCoord></edgeMap>
            </roi>
            <roi><imageZposition>-10</imageZposition><imageSOP_UID>SOP-HOLE</imageSOP_UID>
              <inclusion>FALSE</inclusion>
              <edgeMap><xCoord>100</xCoord><yCoord>200</yCoord></edgeMap>
            </roi>
          </unblindedReadNodule></readingSession>
        </LidcReadMessage>"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotation.xml"
            path.write_text(xml, encoding="utf-8")
            _, annotations = parse_lidc_xml(path)
        self.assertEqual(len(annotations), 1)
        self.assertEqual(annotations[0]["sop_instance_uid"], "SOP-A")
        self.assertEqual(annotations[0]["roi_count"], 1)
        self.assertAlmostEqual(annotations[0]["pixel_x"], 11.0)
        self.assertAlmostEqual(annotations[0]["pixel_y"], 21.0)

    def test_duplicate_series_prefers_resubmitted_correction(self):
        template = """<LidcReadMessage>
        <SeriesInstanceUid>1.2.5</SeriesInstanceUid>
        <readingSession><unblindedReadNodule><noduleID>{nodule_id}</noduleID>
          <roi><imageZposition>0</imageZposition><imageSOP_UID>SOP</imageSOP_UID>
            <edgeMap><xCoord>1</xCoord><yCoord>2</yCoord></edgeMap>
          </roi>
        </unblindedReadNodule></readingSession></LidcReadMessage>"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "001.xml"
            correction = root / "001-resubmitted-correction.xml"
            old.write_text(template.format(nodule_id="OLD"), encoding="utf-8")
            correction.write_text(template.format(nodule_id="NEW"), encoding="utf-8")
            annotations, records = parse_xml_collection([old, correction])
        self.assertEqual(len(annotations), 1)
        self.assertEqual(annotations.iloc[0]["nodule_id"], "NEW")
        self.assertEqual(
            sorted(record["status"] for record in records),
            ["DUPLICATE_SERIESUID", "OK"],
        )


class LIDCPatchTest(unittest.TestCase):
    def test_slice_resolution_prefers_sop_and_has_world_z_fallback(self):
        image = sitk.GetImageFromArray(np.zeros((3, 4, 4), dtype=np.int16))
        image.SetOrigin((0.0, 0.0, 10.0))
        image.SetSpacing((1.0, 1.0, 2.0))
        index, method, error = resolve_slice_index(
            image, 14.0, "SOP-C", {"SOP-C": 2}
        )
        self.assertEqual((index, method, error), (2, "SOP_UID", 0.0))

        index, method, error = resolve_slice_index(image, 12.1, "MISSING", {})
        self.assertEqual(index, 1)
        self.assertEqual(method, "WORLD_Z_FALLBACK")
        self.assertAlmostEqual(error, 0.1)

    def test_border_patch_has_expected_shape_and_padding(self):
        volume = np.zeros((3, 8, 8), dtype=np.int16)
        patch_array = crop_patch_3x64(volume, (0, 0, 0))
        self.assertEqual(patch_array.shape, (3, 64, 64))
        self.assertEqual(patch_array[0, 0, 0], -1000)

    def test_end_to_end_patch_extraction_uses_sop_uid(self):
        uid = "1.2.6"
        image = sitk.GetImageFromArray(
            np.arange(3 * 80 * 80, dtype=np.int16).reshape(3, 80, 80)
        )
        image.SetOrigin((1.0, 2.0, 3.0))
        annotations = pd.DataFrame(
            [
                {
                    "seriesuid": uid,
                    "reader_index": 0,
                    "nodule_id": "N1",
                    "pixel_x": 40.0,
                    "pixel_y": 40.0,
                    "world_z": 4.0,
                    "sop_instance_uid": "SOP-B",
                    "xml_file": "annotation.xml",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "lidc_external.read_dicom_series_with_sop",
            return_value=(image, {"SOP-B": 1}),
        ):
            metadata, records = extract_external_patches(
                annotations,
                {uid: ["unused.dcm"]},
                {uid},
                Path(directory),
            )
            saved = np.load(Path(directory) / metadata.iloc[0]["patch_file"])
        self.assertEqual(saved.shape, (3, 64, 64))
        self.assertTrue(np.isfinite(saved).all())
        self.assertGreaterEqual(float(saved.min()), 0.0)
        self.assertLessEqual(float(saved.max()), 1.0)
        self.assertEqual(metadata.iloc[0]["slice_alignment"], "SOP_UID")
        self.assertEqual(records[0]["status"], "EXTRACTED")


if __name__ == "__main__":
    unittest.main()
