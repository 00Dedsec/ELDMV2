import os
import json
import jieba

frequency = {}

path_list = [
    "..\data\LECARD\stage1",
     "..\data\LECARD\stage1_cutted"
]

dir_list = ['train','valid','test']

def cut(s, stopwords):
    arr = list(jieba.cut(s, cut_all=False))
    tem = " ".join(arr).split()
    arr = [i for i in tem if not i in stopwords]
    for word in arr:
        if not (word in frequency):
            frequency[word] = 0
        frequency[word] += 1
    return arr


if __name__ == "__main__":
    print('begin...')
    result = {}
    with open( 'stopword.txt', 'r') as g:
        words = g.readlines()
    stopwords = [i.strip() for i in words]
    stopwords.extend(['.','（','）','-'])

    input_path = path_list[0]
    output_path = path_list[1]

    os.makedirs(output_path, exist_ok=True)
    for dir in dir_list:
        os.makedirs(os.path.join(output_path, dir), exist_ok=True)
        f = open(os.path.join(input_path, dir, 'query.json'), "r", encoding='utf8')
        for line in f:
            data = []
            x = json.loads(line)
            x['q'] = cut(x['q'], stopwords)
            ridx = x['ridx']
            data.append(x)
            # 处理候选文件
            os.makedirs(os.path.join(output_path, dir, 'candidates' ,str(ridx)), exist_ok=True)
            candidate_filename_list = os.listdir(os.path.join(input_path, dir, 'candidates', str(ridx)))
            for candidate_file_name in candidate_filename_list:
                f_ = open(os.path.join(input_path, dir, 'candidates', str(ridx), candidate_file_name), encoding='utf8')
                candidate_json = json.loads(f_.readline(), encoding='utf-8')
                candidate_json['ajjbqk'] = cut(candidate_json['ajjbqk'], stopwords)
                f_out = open(os.path.join(output_path, dir, 'candidates', str(ridx), candidate_file_name), "w", encoding="utf8")
                print(json.dumps(candidate_json, ensure_ascii=False, sort_keys=True), file=f_out)
                f_out.close
        
        f = open(os.path.join(output_path, dir, 'query.json'), "w", encoding='utf8')
        for x in data:
            print(json.dumps(x, ensure_ascii=False, sort_keys=True), file=f)
        f.close

    json.dump(frequency, open("..\data\LECARD\_frequency.txt", "w", encoding="utf8"),
              indent=2,
              ensure_ascii=False)