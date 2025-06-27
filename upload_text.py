import os
import uuid

from pipeline import Qwen3Embedder, client, COLLECTION_NAME, PointStruct


def parse_txt(filepath):
    with open(filepath, "r", encoding="gb2312") as f:
        lines = f.readlines()

    chunks = []
    current_title = None
    current_paragraph = []

    for line in lines:
        line = line.strip()
        if line.startswith("### "):
            # 如果当前有段落缓存，先保存
            if current_title and current_paragraph:
                full_text = f"{current_title}\n" + "\n".join(current_paragraph)
                chunks.append((current_title, full_text))
            # 开始新的标题
            current_title = line.strip("# ").strip()
            current_paragraph = []
        elif line:  # 非空行，加入当前段落
            current_paragraph.append(line)

    # 最后一组也要保存
    if current_title and current_paragraph:
        full_text = f"{current_title}\n" + "\n".join(current_paragraph)
        chunks.append((current_title, full_text))

    return chunks

def main():
    filepath = "corn_diseases_pests_weeds.txt"
    if not os.path.exists(filepath):
        print(f"[Error] 文件不存在: {filepath}")
        return

    embedder = Qwen3Embedder()
    docs = parse_txt(filepath)

    points = []
    for title, paragraph in docs:
        text_vector = embedder.embed(paragraph)
        point = PointStruct(
            id=uuid.uuid4().hex,
            payload={"title": title, "text": paragraph},
            vector={"text": text_vector} 
        )
        points.append(point)

    print(f"上传 {len(points)} 条文本记录至 Qdrant < {COLLECTION_NAME} >...")
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print("上传完成")


if __name__ == "__main__":
    main()