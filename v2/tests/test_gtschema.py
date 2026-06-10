import unittest
from v2.gtschema import validate_entry, merge_agent_output, cross_check_multiset

GOOD = {"coins":[{"cx":424,"cy":477,"r":85,"label":"5cent","sure":True}],
        "tilt":"none","note":""}

class TestSchema(unittest.TestCase):
    def test_validate_ok(self):
        self.assertEqual(validate_entry("image_2.jpg", GOOD), [])
    def test_validate_bad_label_and_tilt(self):
        bad = {"coins":[{"cx":1,"cy":1,"r":10,"label":"3cent","sure":True}],
               "tilt":"verystrong","note":""}
        errs = validate_entry("image_2.jpg", bad)
        self.assertEqual(len(errs), 2)
    def test_merge_agent_output(self):
        seeds = [{"i":0,"cx":100,"cy":100,"r":50},{"i":1,"cx":300,"cy":300,"r":60}]
        agent = {"image":"image_2.jpg","tilt":"mild",
                 "seeds":[{"i":0,"label":"5cent","sure":True,"circle_ok":True},
                          {"i":1,"label":"FP","sure":True,"circle_ok":True}],
                 "extra":[{"cx":500,"cy":500,"r":55,"label":"1euro","sure":False}]}
        entry = merge_agent_output(agent, seeds)
        self.assertEqual(len(entry["coins"]), 2)          # seed FP escluso, extra incluso
        self.assertEqual(entry["coins"][1]["label"], "1euro")
        self.assertFalse(entry["coins"][1]["sure"])
    def test_cross_check(self):
        entry = {"coins":[{"cx":1,"cy":1,"r":9,"label":"5cent","sure":True}],
                 "tilt":"none","note":""}
        out = cross_check_multiset(entry, ["1cent"])       # GT utente dice 1cent
        self.assertFalse(out["coins"][0]["sure"])          # mismatch → sure:false
        self.assertIn("mismatch", out["note"])

if __name__ == "__main__":
    unittest.main()
