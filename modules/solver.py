import cv2
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application
)
from PIL import Image
import base64, io, re
from groq import Groq


GROQ_API_KEY = ""


class MathSolver:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        print("✅ Groq vision solver ready.")

    def _preprocess(self, bgr_image):
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        inverted = cv2.bitwise_not(gray)

        # Crop tight around content
        coords = cv2.findNonZero(cv2.bitwise_not(inverted))
        if coords is not None:
            bx, by, bw, bh = cv2.boundingRect(coords)
            pad = 40
            x1 = max(0, bx - pad)
            y1 = max(0, by - pad)
            x2 = min(inverted.shape[1], bx + bw + pad)
            y2 = min(inverted.shape[0], by + bh + pad)
            inverted = inverted[y1:y2, x1:x2]

        # Upscale
        h, w = inverted.shape
        if h < 300:
            scale = 300 / h
            inverted = cv2.resize(
                inverted,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_CUBIC
            )

        inverted = cv2.copyMakeBorder(
            inverted, 60, 60, 60, 60,
            cv2.BORDER_CONSTANT, value=255
        )
        _, clean = cv2.threshold(inverted, 200, 255, cv2.THRESH_BINARY)
        return clean

    def _to_base64(self, gray_np):
        pil = Image.fromarray(gray_np).convert("RGB")
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def recognize_and_solve(self, bgr_image):
        processed = self._preprocess(bgr_image)
        cv2.imwrite("debug_canvas.png", bgr_image)
        cv2.imwrite("debug_processed.png", processed)
        print("💾 Saved debug images")

        b64 = self._to_base64(processed)

        print("🔍 Sending to Groq (Llama 4 vision)...")
        try:
            response = self.client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": (
                                "This image contains a handwritten math expression. "
                                "Read it carefully and reply with ONLY the math expression "
                                "in plain text — nothing else, no explanation, no punctuation. "
                                "Use these rules: "
                                "* for multiply, / for divide, ** for power, = for equals. "
                                "Examples of good replies: 5+6  or  3*4  or  10/2  or  x+5=10  or  2**3"
                            )
                        }
                    ]
                }],
                max_tokens=64,
                temperature=0.0   # deterministic — important for math
            )
            raw = response.choices[0].message.content.strip()
            print(f"Groq output:  '{raw}'")

        except Exception as e:
            return f"API Error: {e}", None

        cleaned = self._clean(raw)
        print(f"Cleaned:      '{cleaned}'")
        result = self._evaluate(cleaned)
        return cleaned, result


    def _clean(self, text):
        # Keep only valid math characters
        text = text.strip().replace(" ", "")
        text = re.sub(r'[^0-9+\-*/=().x**]', '', text)
        text = text.replace("×", "*").replace("÷", "/")
        text = text.replace("²", "**2").replace("³", "**3")
        # Only replace x between digits as multiply
        text = re.sub(r'(?<=\d)[xX](?=\d)', '*', text)
        return text

    def _evaluate(self, expr_text):
        try:
            transformations = (
                standard_transformations
                + (implicit_multiplication_application,)
            )

            if "=" in expr_text:
                lhs, rhs = expr_text.split("=", 1)
                lhs_e = parse_expr(lhs, transformations=transformations)
                rhs_e = parse_expr(rhs, transformations=transformations)
                eq = sp.Eq(lhs_e, rhs_e)
                syms = list(eq.free_symbols)
                if syms:
                    sol = sp.solve(eq, syms[0])
                    return f"{syms[0]} = {sol}"
                return str(sp.simplify(lhs_e - rhs_e))

            expr = parse_expr(expr_text, transformations=transformations)
            simplified = sp.simplify(expr)
            try:
                numeric = float(simplified.evalf())
                if numeric == int(numeric):
                    return str(int(numeric))
                return f"{numeric:.4f}"
            except Exception:
                return str(simplified)

        except Exception as e:
            print(f"Parse error: {e}")
            return f"Could not evaluate: {expr_text}"