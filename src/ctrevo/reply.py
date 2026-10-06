"""Decode provider JSON, repairing only unambiguous terminal container framing."""

import json


def decode_reply(text):
    try:
        return json.loads(text), False
    except json.JSONDecodeError as error:
        if error.msg == 'Extra data':
            value, end = json.JSONDecoder().raw_decode(text.lstrip())
            tail = text.lstrip()[end:].strip()
            if isinstance(value, dict) and tail and all(c in '}] \t\r\n' for c in tail):
                return value, True
            raise
        if error.pos != len(text.rstrip()):
            raise
        stack, quoted, escaped = [], False, False
        for char in text.rstrip():
            if quoted:
                if escaped:
                    escaped = False
                elif char == '\\':
                    escaped = True
                elif char == '"':
                    quoted = False
            elif char == '"':
                quoted = True
            elif char in '{[':
                stack.append('}' if char == '{' else ']')
            elif char in '}]':
                if not stack or stack.pop() != char:
                    raise error
        if quoted or not stack:
            raise error
        # No values, commas, keys, string characters or source code are synthesized.
        return json.loads(text.rstrip() + ''.join(reversed(stack))), True
