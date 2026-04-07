import math
import os
import re

from tqdm import tqdm

from metrics import Calculator
from utils import load_jsonl, save_jsonl


class W4E2(object):
    def __init__(self, args):
        self.args = args
        self.subset = args.subset
        self.model_path = args.model_path
        self.data_dir = f"{args.dataset_dir}/{self.subset}"
        self.gt_dir = args.gt_dir
        self.save_dir = f"{args.save_dir}/{self.subset}/{args.model_path.split('/')[-1]}"
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.gt_file = load_jsonl(f"{self.gt_dir}/{self.subset}.jsonl")
        self.question_type = {
            'MCQ': ['act', 'sub', 'obj', 'loc'], 
            'OEQ': ['eff_sub', 'eff_obj'], 
            'ALL': ['act', 'sub', 'obj', 'loc', 'eff_sub', 'eff_obj']
        }
        
    def inference(self):
        video_list = sorted(os.listdir(self.data_dir), key=lambda x: (int(x.split('-')[0]), '-'.join(x.split('-')[1:])))
        num_processed = len(load_jsonl(f"{self.save_dir}/answers.jsonl")) if os.path.isfile(f"{self.save_dir}/answers.jsonl") else 0
        
        from models import get_model
        model = get_model(self.model_path, **{"subset": self.subset, "api_key":self.args.api_key})
        
        for idx, video in enumerate(tqdm(video_list, dynamic_ncols=True)):
            gt = self.gt_file[idx]
            video_path = f'{self.data_dir}/{video}'
            assert gt["video"] == video
            candidates = {k: "; ".join([f"{idx}. {i}" for idx, i in enumerate(gt[k]['cand'], start=1)]) for k in self.question_type['MCQ']}
            
            questions = {
                "act":  f"Candidate behaviors: {candidates['act']}.\n\n"
                        "Which one of these candidate behaviors is shown in the video?\n\n"
                        "Your answer should follow this format strictly: <The index of the candidate you choose>",
                
                "sub":  f"Candidate terms: {candidates['sub']}.\n\n"
                        "Which one of the candidate terms best refer to the performer of this action?\n\n"
                        "Your answer should follow this format strictly: <The index of the candidate you choose>",
                            
                "obj":  f"Candidate objects: {candidates['obj']}.\n\n"
                        "Which one of the candidate objects does the performer primarily interact with?"
                        "Your answer should follow this format strictly: <The index of the candidate you choose>",
                        
                "loc":  f"Candidate locations: {candidates['loc']}.\n\n"
                        "In which one of these candidate locations does the action in the video take place?\n\n"
                        "Your answer should follow this format strictly: <The index of the candidate you choose>",
                    
                "eff_sub":  "What effect might this action have on the performer?\n\n"
                            "Your answer should be concise and comprehensive.",
                    
                "eff_obj":  "What effect might this action have on the object with which the performer interacts?\n\n"
                            "Your answer should be concise and comprehensive.\n\n"
                            "If the performer of the action in the video does not interact with any object, your answer should be <None>.",
            }
            
            if idx < num_processed:
                continue
            
            answers = {'video': video}
            for k, q in questions.items():
                response = model.converse(video_path, q)
                answers[k] = response
            
            # for i in answers.keys():
            #     if i != 'video':
            #         save_jsonl([{video: answers[i]}], f"{self.save_dir}/{i}.jsonl", mode='a')
                    
            save_jsonl([answers], f"{self.save_dir}/answers.jsonl", mode='a')
            
    def evaluate(self):
        video_list = sorted(os.listdir(self.data_dir), key=lambda x: (int(x.split('-')[0]), '-'.join(x.split('-')[1:])))
        answer_file = load_jsonl(f"{self.save_dir}/answers.jsonl")
        sim_calculator = Calculator()
        
        for idx, item in enumerate(self.gt_file):
            assert item['video'] == answer_file[idx]['video']
        
        gt = {k: [i[k]['gt'] if k in self.question_type['MCQ'] else i[k] for i in self.gt_file] for k in self.question_type['ALL']}
        answer = {k: [answer_file[i][k] for i in range(len(answer_file))] for k in self.question_type['ALL']}
        
        acc = {}
        for k in answer.keys():
            if k in self.question_type['MCQ']:
                answer[k] = [int(re.sub(r'\D', '', i)) if re.sub(r'\D', '', i) else 0 for i in answer[k]]
                acc[k] = sum([answer[k][idx] == gt[k][idx] for idx in range(len(gt[k]))]) / len(self.gt_file)
            else:
                scores = []
                for idx, ans in enumerate(tqdm(answer[k])):
                    ans = re.sub(r'<|>', '', ans)
                    s = sim_calculator.calculate(ans, gt[k][idx])
                    scores.append(s)
                acc[k] = sum(scores) / len(self.gt_file)
                
        gmean = math.pow(acc['act'] * acc['sub'] * acc['obj'] * acc['loc'], 1/4)
        acc['eff_sub'] = acc['eff_sub'] * gmean
        acc['eff_obj'] = acc['eff_obj'] * gmean
        
        print(f"Accuracy of each question: {[round(a*100, 2) for a in acc.values()]}\n")
        