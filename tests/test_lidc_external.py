import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lidc_external import parse_lidc_xml


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


if __name__ == "__main__":
    unittest.main()
