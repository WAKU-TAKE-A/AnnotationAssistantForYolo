import os
import shutil
import subprocess
import tkinter as tk
from tkinter import filedialog
import random
import ttkbootstrap as tb
import threading

from ultralytics import YOLO

# ここでPython実行コマンドを指定してください
PYTHON_CMD = "C:/WinPython3.10dot/python-3.10.11.amd64/python.exe"

class AnnotationFileMoverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AnnotationPrePostHelper")
        self.root.geometry("600x400")

        # ttkbootstrapスタイル指定
        self.style = tb.Style(theme="flatly")

        # フォルダパスの保存
        self.selected_folder = None

        # ボタンフレーム（縦に並べる）
        button_frame = tb.Frame(root)
        button_frame.pack(pady=10, fill='y', padx=10, side="left")

        # 各ボタンとテキストボックス
        self.btn_select_folder = tb.Button(
            button_frame,
            text="プロジェクトフォルダの選択",
            bootstyle="primary",
            command=self.select_folder
        )
        self.btn_select_folder.pack(side="top", fill='x', pady=5)

        self.btn_predict = tb.Button(
            button_frame,
            text="アノテーション前の推論処理",
            bootstyle="secondary",
            command=self.predict_yolo,
            state="disabled"
        )
        self.btn_predict.pack(side="top", fill='x', pady=5)

        self.btn_pre_move = tb.Button(
            button_frame,
            text="アノテーション前のファイル移動",
            bootstyle="info",
            command=self.pre_annotation_move,
            state="disabled"
        )
        self.btn_pre_move.pack(side="top", fill='x', pady=5)

        self.btn_labelimg = tb.Button(
            button_frame,
            text="labelImgの起動",
            bootstyle="success",
            command=self.run_labelimg,
            state="disabled"
        )
        self.btn_labelimg.pack(side="top", fill='x', pady=5)

        self.btn_post_move = tb.Button(
            button_frame,
            text="アノテーション後のファイル移動",
            bootstyle="info",
            command=self.post_annotation_move,
            state="disabled"
        )
        self.btn_post_move.pack(side="top", fill='x', pady=5)

        self.btn_change_label_id = tb.Button(
            button_frame,
            text="ラベルIDの変更",
            bootstyle="warning",
            command=self.change_label_ids,
            state="disabled"
        )
        self.btn_change_label_id.pack(side="top", fill='x', pady=5)

        self.btn_all_distribute = tb.Button(
            button_frame,
            text="allから振り分け",
            bootstyle="warning",
            command=self.distribute_all_files,
            state="disabled"
        )
        self.btn_all_distribute.pack(side="top", fill='x', pady=5)

        self.ratio_text_var = tk.StringVar(value="8:1:1")
        self.entry_ratio = tb.Entry(button_frame, textvariable=self.ratio_text_var)
        self.entry_ratio.pack(side="top", fill='x', pady=5)

        self.btn_train = tb.Button(
            button_frame,
            text="追加学習",
            bootstyle="success",
            command=self.run_train_script,
            state="disabled"
        )
        self.btn_train.pack(side="top", fill='x', pady=5)

        # メッセージ表示テキストはボタンの右側に配置（横に広げる）
        self.message_text = tk.Text(root, height=20, state="disabled")
        self.message_text.pack(fill="both", expand=True, padx=10, pady=10, side="left")

    def log_message(self, message):
        # UIスレッドで実行するように呼び出しを調整
        def append():
            self.message_text.config(state="normal")
            self.message_text.insert(tk.END, message + "\n")
            self.message_text.see(tk.END)
            self.message_text.config(state="disabled")

        self.root.after(0, append)

    def clear_message(self):
        # UIスレッドで実行するように呼び出しを調整
        def clear():
            self.message_text.config(state="normal")
            self.message_text.delete(1.0, tk.END)
            self.message_text.config(state="disabled")

        self.root.after(0, clear)

    def select_folder(self):
        folder = filedialog.askdirectory(title="メインフォルダを選択してください")
        if not folder:
            return

        self.clear_message()
        self.selected_folder = folder
        self.log_message(f"選択フォルダ: {folder}")

        error_messages = []

        # images/train, images/valid, images/test, images/all 存在確認
        # labels/train, labels/valid, labels/test, labels/all 存在確認
        required_subdirs = ["train", "valid", "test", "all"]
        for d in required_subdirs:
            img_path = os.path.join(folder, "images", d)
            lbl_path = os.path.join(folder, "labels", d)
            if not os.path.isdir(img_path):
                error_messages.append(f"フォルダが存在しません: images/{d}")
            if not os.path.isdir(lbl_path):
                error_messages.append(f"フォルダが存在しません: labels/{d}")

        # classes.txtは選択フォルダ直下にあるか調べる
        classes_txt_path = os.path.join(folder, "classes.txt")
        if not os.path.isfile(classes_txt_path):
            error_messages.append(f"選択フォルダ直下に classes.txt がありません。")

        # train.pyとdata.yamlも確認（なくても警告程度）
        train_py_path = os.path.join(folder, "train.py")
        if not os.path.isfile(train_py_path):
            self.log_message("警告: 選択フォルダ直下に train.py が見つかりません。")

        data_yaml_path = os.path.join(folder, "data.yaml")
        if not os.path.isfile(data_yaml_path):
            self.log_message("警告: 選択フォルダ直下に data.yaml が見つかりません。")

        if error_messages:
            self.log_message("エラー：以下の項目を確認してください。")
            for msg in error_messages:
                self.log_message(" - " + msg)
            self.btn_pre_move.configure(state="disabled")
            self.btn_post_move.configure(state="disabled")
            self.btn_change_label_id.configure(state="disabled")
            self.btn_train.configure(state="disabled")
            self.btn_labelimg.configure(state="disabled")
            self.btn_all_distribute.configure(state="disabled")
            self.btn_predict.configure(state="disabled")
            return
        else:
            self.log_message("すべてのフォルダ・ファイル構造を確認しました。OKです。")
            self.btn_pre_move.configure(state="normal")
            self.btn_post_move.configure(state="normal")
            self.btn_change_label_id.configure(state="normal")
            self.btn_train.configure(state="normal")
            self.btn_labelimg.configure(state="normal")
            self.btn_all_distribute.configure(state="normal")
            self.btn_predict.configure(state="normal")

    def pre_annotation_move(self):
        self.clear_message()
        if not self.selected_folder:
            self.log_message("フォルダが選択されていません。")
            return

        self.log_message("アノテーション前のファイル移動を開始します。")
        required_subdirs = ["train", "valid", "test", "all"]

        for d in required_subdirs:
            images_path = os.path.join(self.selected_folder, "images", d)
            labels_path = os.path.join(self.selected_folder, "labels", d)

            if not os.path.isdir(labels_path) or not os.path.isdir(images_path):
                self.log_message(f"{d}のlabelsまたはimagesフォルダが見つかりません。スキップします。")
                continue

            # labels内のtxtをimagesへ移動
            moved_count = 0
            for f in os.listdir(labels_path):
                if f.lower().endswith(".txt"):
                    src = os.path.join(labels_path, f)
                    dst = os.path.join(images_path, f)
                    try:
                        shutil.move(src, dst)
                        moved_count += 1
                    except Exception as e:
                        self.log_message(f"{d} - {f} の移動に失敗: {e}")

            self.log_message(f"{d}：labels から images へ {moved_count} ファイルを移動しました。")

            # classes.txt を imagesにコピー（選択フォルダ直下からコピー）
            src_classes = os.path.join(self.selected_folder, "classes.txt")
            dst_classes = os.path.join(images_path, "classes.txt")
            if os.path.isfile(src_classes):
                try:
                    shutil.copy2(src_classes, dst_classes)
                    self.log_message(f"{d}：classes.txt を images フォルダにコピーしました。")
                except Exception as e:
                    self.log_message(f"{d}：classes.txt のコピーに失敗: {e}")
            else:
                self.log_message(f"{d}：classes.txt のコピー元が見つかりませんでした。")

        self.log_message("アノテーション前のファイル移動が完了しました。")

    def post_annotation_move(self):
        self.clear_message()
        if not self.selected_folder:
            self.log_message("フォルダが選択されていません。")
            return

        self.log_message("アノテーション後のファイル移動を開始します。")
        required_subdirs = ["train", "valid", "test", "all"]

        for d in required_subdirs:
            images_path = os.path.join(self.selected_folder, "images", d)
            labels_path = os.path.join(self.selected_folder, "labels", d)

            if not os.path.isdir(images_path) or not os.path.isdir(labels_path):
                self.log_message(f"{d}のimagesまたはlabelsフォルダが見つかりません。スキップします。")
                continue

            # images内のclasses.txtを削除
            classes_txt_path = os.path.join(images_path, "classes.txt")
            if os.path.isfile(classes_txt_path):
                try:
                    os.remove(classes_txt_path)
                    self.log_message(f"{d}：images フォルダ内の classes.txt を削除しました。")
                except Exception as e:
                    self.log_message(f"{d}：classes.txt の削除に失敗: {e}")

            # images内のtxtをlabelsに移動
            moved_count = 0
            for f in os.listdir(images_path):
                if f.lower().endswith(".txt"):
                    src = os.path.join(images_path, f)
                    dst = os.path.join(labels_path, f)
                    try:
                        shutil.move(src, dst)
                        moved_count += 1
                    except Exception as e:
                        self.log_message(f"{d} - {f} の移動に失敗: {e}")

            self.log_message(f"{d}：images から labels へ {moved_count} ファイルを移動しました。")

        self.log_message("アノテーション後のファイル移動が完了しました。")

    def change_label_ids(self):
        self.clear_message()
        if not self.selected_folder:
            self.log_message("フォルダが選択されていません。")
            return

        self.log_message("変更するラベルIDのCSVファイルを選択してください。")

        csv_path = filedialog.askopenfilename(
            title="ラベルID変更用CSVファイルを選択してください",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not csv_path:
            self.log_message("CSVファイルが選択されませんでした。処理を中止します。")
            return

        import csv

        # CSVから辞書作成
        try:
            mapping = {}
            with open(csv_path, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                if 'before' not in reader.fieldnames or 'after' not in reader.fieldnames:
                    self.log_message("CSVに 'before' および 'after' 列が存在しません。")
                    return

                for row in reader:
                    b = row['before'].strip()
                    a = row['after'].strip()
                    if b == '' or a == '':
                        continue
                    mapping[b] = a

            if not mapping:
                self.log_message("CSVから有効な変更マッピングが見つかりません。")
                return

            self.log_message(f"CSVから {len(mapping)} 件のIDマッピングを読み込みました。")

        except Exception as e:
            self.log_message(f"CSV読み込み時にエラーが発生しました: {e}")
            return

        required_subdirs = ["train", "valid", "test", "all"]
        total_files = 0
        total_lines_changed = 0

        for d in required_subdirs:
            for folder_type, folder_name in [("images", "images"), ("labels", "labels")]:
                folder_path = os.path.join(self.selected_folder, folder_name, d)
                if not os.path.isdir(folder_path):
                    self.log_message(f"{folder_name}/{d} フォルダが存在しません。スキップします。")
                    continue

                txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]

                for txt_file in txt_files:
                    file_path = os.path.join(folder_path, txt_file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            lines = f.readlines()

                        new_lines = []
                        changed_count = 0
                        for line in lines:
                            parts = line.strip().split(maxsplit=1)
                            if len(parts) == 0:
                                new_lines.append(line)
                                continue
                            id_ = parts[0]
                            rest = parts[1] if len(parts) > 1 else ""
                            # 置換が必要ならIDを変更
                            if id_ in mapping:
                                new_id = mapping[id_]
                                new_line = new_id + " " + rest + "\n"
                                changed_count += 1
                            else:
                                new_line = line
                            new_lines.append(new_line)

                        if changed_count > 0:
                            with open(file_path, "w", encoding="utf-8") as f:
                                f.writelines(new_lines)
                            self.log_message(f"{folder_name}/{d}/{txt_file} ： {changed_count} 行のIDを変更しました。")
                            total_lines_changed += changed_count

                        total_files += 1

                    except Exception as e:
                        self.log_message(f"{folder_name}/{d}/{txt_file} の処理中にエラー: {e}")

        self.log_message(f"処理が完了しました。合計 {total_files} ファイル処理、変更行数 {total_lines_changed} 行。")

    # ⑤ allから振り分け処理（画像ファイルをランダムシャッフルし、指定比率でtrain/valid/testにコピー）
    def distribute_all_files(self):
        self.clear_message()
        if not self.selected_folder:
            self.log_message("フォルダが選択されていません。")
            return

        # 比率テキスト取得
        ratio_str = self.ratio_text_var.get().strip()
        try:
            parts = ratio_str.split(":")
            if len(parts) != 3:
                raise ValueError("比率は「数字:数字:数字」の形式で入力してください。")
            ratio_nums = list(map(int, parts))
            if any(n < 0 for n in ratio_nums):
                raise ValueError("比率には負の値は使えません。")
            total = sum(ratio_nums)
            if total == 0:
                raise ValueError("比率の合計は0以外でなければなりません。")
        except Exception as e:
            self.log_message(f"比率入力エラー: {e}")
            return

        images_all = os.path.join(self.selected_folder, "images", "all")
        labels_all = os.path.join(self.selected_folder, "labels", "all")

        if not (os.path.isdir(images_all) and os.path.isdir(labels_all)):
            self.log_message("allフォルダ内に images または labels フォルダが存在しません。振り分け処理を中止します。")
            return

        # all/imagesの画像ファイル判定（例として一般的な拡張子）
        img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

        all_image_files = [f for f in os.listdir(images_all)
                           if os.path.splitext(f)[1].lower() in img_exts]

        if not all_image_files:
            self.log_message("all/imagesフォルダに画像ファイルが見つかりません。")
            return

        # ランダムシャッフル
        random.shuffle(all_image_files)

        total_files = len(all_image_files)
        train_count = (total_files * ratio_nums[0]) // total
        valid_count = (total_files * ratio_nums[1]) // total
        test_count = (total_files * ratio_nums[2]) // total
        assigned_sum = train_count + valid_count + test_count
        if assigned_sum < total_files:
            train_count += total_files - assigned_sum  # 端数はtrainに

        self.log_message(f"振り分け割合: train={train_count}, valid={valid_count}, test={test_count} (合計 {total_files})")

        # 対象フォルダパス辞書
        target_dirs = {
            'train': os.path.join(self.selected_folder, 'train'),
            'valid': os.path.join(self.selected_folder, 'valid'),
            'test': os.path.join(self.selected_folder, 'test'),
        }

        # images と labels のパスを持つ辞書も作成
        target_paths = {}
        for key in ['train', 'valid', 'test']:
            target_paths[key] = {
                "images": os.path.join(self.selected_folder, "images", key),
                "labels": os.path.join(self.selected_folder, "labels", key),
            }
            # 必要ならターゲットフォルダがない場合は作成する
            for subfolder in ["images", "labels"]:
                p = target_paths[key][subfolder]
                if not os.path.isdir(p):
                    os.makedirs(p)

        def copy_file(src, dst):
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                self.log_message(f"コピー失敗: {src} -> {dst} : {e}")

        # 振り分け実行
        idx = 0
        for key, count in zip(['train', 'valid', 'test'], [train_count, valid_count, test_count]):
            self.log_message(f"{key} に {count} ファイルを振り分けます。")
            for _ in range(count):
                if idx >= total_files:
                    break
                img_file = all_image_files[idx]
                idx += 1

                # 画像のコピー先
                src_img_path = os.path.join(images_all, img_file)
                dst_img_path = os.path.join(target_paths[key]["images"], img_file)

                # テキストファイル対応
                base_name = os.path.splitext(img_file)[0]
                possible_txt = base_name + ".txt"
                src_txt_path = os.path.join(labels_all, possible_txt)
                dst_txt_path = os.path.join(target_paths[key]["labels"], possible_txt)

                copy_file(src_img_path, dst_img_path)
                if os.path.isfile(src_txt_path):
                    copy_file(src_txt_path, dst_txt_path)
                else:
                    self.log_message(f"対応するテキストファイルが見つかりません: {possible_txt} （画像: {img_file}）")

        self.log_message("all から train/valid/test への振り分け処理が完了しました。")

    def run_train_script(self):
        # ここを別スレッドで実行してUIの応答性を保つ
        def worker():
            self.clear_message()
            if not self.selected_folder:
                self.log_message("フォルダが選択されていません。")
                return

            train_py_path = os.path.join(self.selected_folder, "train.py")
            if not os.path.isfile(train_py_path):
                self.log_message("train.py が選択フォルダ直下に見つかりません。")
                return

            self.log_message("train.py の実行を開始します。")

            try:
                process = subprocess.Popen(
                    [PYTHON_CMD, "-u", train_py_path],
                    cwd=self.selected_folder,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )

                while True:
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break
                    if line:
                        self.log_message(line.strip())

                process.wait()
                if process.returncode == 0:
                    self.log_message("train.py の実行が正常に完了しました。")
                else:
                    self.log_message(f"train.py は異常終了しました。終了コード: {process.returncode}")

            except Exception as e:
                self.log_message(f"train.py の実行中にエラーが発生しました: {e}")

        threading.Thread(target=worker, daemon=True).start()

    def run_labelimg(self):
        self.clear_message()
        # PYTHON_CMDのフォルダパス取得
        python_dir = os.path.dirname(PYTHON_CMD)
        labelimg_path = os.path.join(python_dir, "Scripts", "labelImg.exe")

        if not os.path.isfile(labelimg_path):
            self.log_message(f"labelImg.exe が見つかりません。パスを確認してください:\n{labelimg_path}")
            return

        classes_txt_path = os.path.join(self.selected_folder, "classes.txt")
        if not os.path.isfile(classes_txt_path):
            self.log_message("classes.txt が選択フォルダにありません。labelImgを起動しますが、ラベル読み込みに失敗する可能性があります。")

        self.log_message("labelImg.exe を起動します。")

        try:
            # 作業ディレクトリをPYTHON_CMDのあるフォルダに変更して起動
            subprocess.Popen(
                [labelimg_path, "class_file", classes_txt_path],
                cwd=python_dir,
                shell=True  # Windowsの実行ファイルなのでshell=True推奨
            )
            self.log_message("labelImg.exe を起動しました。")
        except Exception as e:
            self.log_message(f"labelImg の起動に失敗しました: {e}")

    def predict_yolo(self):
        # スレッドで動かす
        def worker():
            self.clear_message()
            if not self.selected_folder:
                self.log_message("フォルダが選択されていません。")
                return

            # ptファイル選択
            pt_path = filedialog.askopenfilename(
                title="モデルファイル (.pt) を選択してください",
                filetypes=[("PyTorch Model", "*.pt"), ("All files", "*.*")]
            )
            if not pt_path:
                self.log_message("モデルファイルが選択されませんでした。処理を中止します。")
                return

            self.log_message(f"モデルファイルを読み込みます: {pt_path}")

            try:
                model = YOLO(pt_path)
            except Exception as e:
                self.log_message(f"モデルの読み込みエラー: {e}")
                return

            required_subdirs = ["train", "valid", "test", "all"]
            total_images = 0
            predicted_images = 0

            for d in required_subdirs:
                images_path = os.path.join(self.selected_folder, "images", d)
                labels_path = os.path.join(self.selected_folder, "labels", d)

                if not (os.path.isdir(images_path) and os.path.isdir(labels_path)):
                    self.log_message(f"{d} フォルダの images または labels が見つからずスキップします。")
                    continue

                img_files = [f for f in os.listdir(images_path)
                             if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif"}]

                for img_file in img_files:
                    total_images += 1
                    label_file = os.path.splitext(img_file)[0] + ".txt"
                    label_path = os.path.join(labels_path, label_file)

                    # labelsが既にあればスキップ
                    if os.path.isfile(label_path):
                        #self.log_message(f"{d}: {img_file} はラベルが存在するためスキップします。")
                        continue

                    img_path = os.path.join(images_path, img_file)

                    try:
                        # 推論実行
                        results = model.predict(source=img_path, save=False, verbose=False)

                        # 推論結果は results[0].boxes にある
                        result = results[0]
                        boxes = result.boxes

                        if boxes is None or len(boxes) == 0:
                            self.log_message(f"{d}: {img_file} は推論で物体検出なし。空ファイルを作成します。")
                            with open(label_path, "w", encoding="utf-8") as f_out:
                                pass  # 空ファイル
                        else:
                            with open(label_path, "w", encoding="utf-8") as f_out:
                                # boxes.cls と boxes.xywhn が利用可能
                                cls_list = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else boxes.cls
                                yolo_list = boxes.xywhn.cpu().numpy() if hasattr(boxes.xywhn, "cpu") else boxes.xywhn
                                for cls_, box in zip(cls_list, yolo_list):
                                    cls_i = int(cls_)
                                    x_c, y_c, w, h = box
                                    f_out.write(f"{cls_i} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

                        predicted_images += 1
                        self.log_message(f"{d}: {img_file} の推論結果を保存しました。")

                    except Exception as e:
                        self.log_message(f"{d}: {img_file} 処理中にエラー: {e}")

            self.log_message(f"推論処理が完了しました。処理画像数: {predicted_images} / 総画像数: {total_images}")

        threading.Thread(target=worker, daemon=True).start()


def main():
    root = tb.Window()
    app = AnnotationFileMoverApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()