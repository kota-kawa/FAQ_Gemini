import json
import os
import uuid
from datetime import datetime

EVAL_FILE = "evaluation/evaluation_data.json"

def _load_data():
    if not os.path.exists(EVAL_FILE):
        return {"tasks": [], "results": []}
    try:
        with open(EVAL_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"tasks": [], "results": []}

def _save_data(data):
    os.makedirs(os.path.dirname(EVAL_FILE), exist_ok=True)
    with open(EVAL_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_tasks():
    data = _load_data()
    return data.get("tasks", [])

def add_tasks(new_tasks):
    data = _load_data()
    tasks = data.get("tasks", [])
    for t in new_tasks:
        # Create a simple ID if not present
        if "id" not in t:
            t["id"] = str(uuid.uuid4())
        tasks.append(t)
    data["tasks"] = tasks
    _save_data(data)

def clear_data():
    _save_data({"tasks": [], "results": []})

def add_result(task_id, model, question, expected_answer, actual_answer, status="pending"):
    data = _load_data()
    results = data.get("results", [])
    result = {
        "id": str(uuid.uuid4()),
        "task_id": task_id,
        "model": model,
        "question": question,
        "expected_answer": expected_answer,
        "actual_answer": actual_answer,
        "status": status,
        "timestamp": datetime.now().isoformat()
    }
    results.append(result)
    data["results"] = results
    _save_data(data)
    return result

def get_results():
    data = _load_data()
    return data.get("results", [])

def update_result_status(result_id, status):
    data = _load_data()
    results = data.get("results", [])
    for r in results:
        if r["id"] == result_id:
            r["status"] = status
            break
    data["results"] = results
    _save_data(data)

SAMPLE_TASKS = [
    {
        "question": "[家電] アイロンがけの後、戻りジワを防ぐためにすぐに行うべきことは何ですか？",
        "expected_answer": "ハンガーで冷ますこと"
    },
    {
        "question": "[料理] アボカドの食べ頃を見分ける際、皮の色以外に確認すべきポイントはどこですか？",
        "expected_answer": "指で軽く押して弾力を感じるくらい（がベスト）\nヘタが取れそうなもの（も熟している）"
    },
    {
        "question": "[金融] 新NISAにおいて、「つみたて投資枠」の年間投資上限額はいくらですか？",
        "expected_answer": "年間120万円（月額10万円）"
    },
    {
        "question": "[メンタル] 怒りの感情のピークは何秒間続くとされていますか？",
        "expected_answer": "6秒"
    },
    {
        "question": "[料理] 鶏胸肉を焼く際、パサつかせずしっとり仕上げるための具体的な下処理方法は？",
        "expected_answer": "削ぎ切りにする\n塩麹かマヨネーズを揉み込んで10分置く\n片栗粉をまぶして弱火で焼く"
    },
    {
        "question": "[メンタル] 不安で胸がざわざわする時に有効な「4-7-8呼吸法」の具体的なやり方は？",
        "expected_answer": "4秒吸って、7秒止めて、8秒かけて吐く"
    },
    {
        "question": "[料理] 今すごく疲れていて包丁を使いたくありません。冷蔵庫に豚バラと白菜があるのですが、これで作れる簡単な料理はありますか？",
        "expected_answer": "「豚バラと白菜の重ね蒸し」（元データでは「豚バラと白菜・大根の重ね蒸し」として紹介）\n調理法：鍋に敷き詰めて酒と鶏ガラスープの素を入れて放置するだけ（大根を入れる場合はピーラーを使えば包丁いらず）"
    },
    {
        "question": "[料理] 週末に作り置きをしたいです。お弁当に入れられて、かつ冷凍保存もできるおかずをいくつか提案してください。",
        "expected_answer": "鶏むね肉の南蛮漬け（冷めても美味しく、酢の効果で保存性が高い）\nひじきと大豆の煮物（お弁当の隙間埋めに最適）\n無限ピーマンの肉味噌炒め（ご飯に乗せるだけで丼になる）"
    },
    {
        "question": "[金融] 老後資金のためにiDeCoとNISAで迷っています。iDeCo特有のメリットと、逆に気をつけなければならない「制限」は何ですか？",
        "expected_answer": "メリット：掛金が全額所得控除になり節税効果が高い\n制限：原則60歳まで引き出せない"
    },
    {
        "question": "[メンタル] 明日大事なプレゼンがあるのに緊張して眠れません。無理に寝ようとして焦ってしまうのですが、どう対処すればいいですか？",
        "expected_answer": "部屋を暗くして「米軍式睡眠法」の筋弛緩を試す\nそれでも眠れない場合は、「横になって目を閉じているだけでも体は8割程度休まる」と割り切る\n（明日の朝、カフェインを摂取するタイミングだけ調整すればパフォーマンスは維持できる）"
    }
]
