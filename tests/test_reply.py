import json
import unittest
from ctrevo.reply import decode_reply


class ReplyTests(unittest.TestCase):
    def test_framing_repairs_preserve_every_value_and_code_character(self):
        value = {'candidate': {'source': 'def f():\n return {"x": "}\\\\"}\n', 'config': {'model': {}}}}
        raw = json.dumps(value)
        self.assertEqual(decode_reply(raw), (value, False))
        self.assertEqual(decode_reply(raw[:-1]), (value, True))
        self.assertEqual(decode_reply(raw + '}'), (value, True))

    def test_ambiguous_or_incomplete_content_is_not_repaired(self):
        for raw in ['{"a":', '{"a":"unterminated', '{"a":1 "b":2}', '{} {}', '{} explanation']:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                decode_reply(raw)
