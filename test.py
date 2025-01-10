import zerorpc

def send_text(text):
    client = zerorpc.Client(timeout=60, heartbeat=60)
    client.connect("tcp://localhost:4242")
    payload = {"text": text}
    client.infer(payload)

if __name__ == "__main__":
    text = """1.机器人不得伤害人类,或坐视人类受到伤害。2.机器人必须服从人类命令,除非命令与第一法则发生冲突。3.在不违背第一或第二法则之下，机器人可以保护自己。
"""
    # text = """机器人不得伤害人类,或坐视人类受到伤害。"""
    send_text(text)
    print("finish sending the text...")