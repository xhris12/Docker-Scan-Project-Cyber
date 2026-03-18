import sys
import random as rd
import requests
from io import BytesIO

import torch
import numpy as np
from PIL import Image
from torchvision import transforms, models
from torchvision.utils import save_image

class ZOOAdam:
    """
    ZOO-ADAM Optimizer for Zeroth Order Stochastic Coordinate Descent.
    """
    def __init__(self, x_init, score_fn, lr=1e-1, betas=(0.9, 0.999), eps=1e-8, h=0.001, box=1):
        self.X = x_init.clone().detach().float()
        self.M = torch.zeros_like(self.X)
        self.u = torch.zeros_like(self.X)
        self.T = torch.zeros_like(self.X, dtype=torch.long)

        self.best_X = self.X.clone()
        self.score_fn = score_fn
        self.eta = lr
        self.beta_1, self.beta_2 = betas

        self.eps = eps
        self.h = h
        self.scores = []

        self.init_lr = lr
        self.C, self.H, self.W = self.X.shape
        self.col = self.row = box

        self.len = self.H / self.col
        self.flag = False

    def step(self):
        def return_Eijk():
            i = rd.randint(0, self.col - 1)
            j = rd.randint(0, self.row - 1)
            c = rd.randint(0, self.C - 1)

            start_i = int(i * self.len)
            end_i = min(int((i + 1) * self.len), self.H)
            start_j = int(j * self.len)
            end_j = min(int((j + 1) * self.len), self.W)

            e_ijk = torch.zeros_like(self.X)
            e_ijk[c, start_i:end_i, start_j:end_j] = 1.0
            return e_ijk, c, start_i, end_i, start_j, end_j

        e_ijk, c, si, ei, sj, ej = return_Eijk()
        idx_block = (c, si, sj)
        
        self.T[e_ijk == 1] += 1
        t = self.T[idx_block].item()

        X_right = (self.X + self.h * e_ijk).clamp(0, 1)
        X_left = (self.X - self.h * e_ijk).clamp(0, 1)
        
        self.best_X = X_right.clone()
        f_right = self.score_fn(X_right)
        
        self.best_X = X_left.clone()
        f_left = self.score_fn(X_left)

        g = (f_right - f_left) / (2 * self.h)

        self.M[c, si:ei, sj:ej] = self.beta_1 * self.M[c, si:ei, sj:ej] + (1 - self.beta_1) * g
        self.u[c, si:ei, sj:ej] = self.beta_2 * self.u[c, si:ei, sj:ej] + (1 - self.beta_2) * g**2

        m_hat = self.M[idx_block] / (1 - self.beta_1 ** t)
        u_hat = self.u[idx_block] / (1 - self.beta_2 ** t)

        delta = self.eta * m_hat / (torch.sqrt(u_hat) + self.eps)

        self.X[c, si:ei, sj:ej] += delta
        self.X[c, si:ei, sj:ej] = self.X[c, si:ei, sj:ej].clamp(0, 1)

    def BlockOptimization(self):
        step = 1
        eta_sched = 0
        while True:
            self.step()
            if step % (3 * self.row * self.col) == 0:
                score_val = self.eval()
                print(f"Score: {score_val:.4f}")
                
                if self.flag:
                    if len(self.scores) > 1 and self.scores[-2] > 0.99 * self.scores[-1]:
                        self.eta *= 0.75
                else:
                    if len(self.scores) > 1 and self.scores[-2] > 0.99 * self.scores[-1]:
                        self.eta *= 0.75
                        eta_sched += 1
                        if eta_sched == 5:
                            break
            step += 1

    def eval(self):
        self.best_X = self.X.clone()
        eval_score = self.score_fn(self.X)
        self.scores.append(eval_score)
        return eval_score

    def get_X(self):
        return self.best_X

    def increase(self):
        if self.row * 2 > self.H:
            self.flag = True
            return
        self.eta = self.init_lr
        self.row *= 2
        self.col *= 2
        self.len = self.H / self.col
        self.M = torch.zeros_like(self.X)
        self.u = torch.zeros_like(self.X)
        self.T = torch.zeros_like(self.X, dtype=torch.long)


def load_target_image(url, size):
    """Κατεβάζει την εικόνα-στόχο και την μετατρέπει σε Tensor."""
    response = requests.get(url)
    if response.status_code == 200:
        auth = Image.open(BytesIO(response.content)).resize((size, size)).convert("RGB")
        transform = transforms.ToTensor()
        return transform(auth)
    else:
        raise Exception(f"Failed to download image from {url}")

def get_similarity_function(task):
    """Επιστρέφει την κατάλληλη συνάρτηση similarity με βάση το επίπεδο δυσκολίας."""
    if task == 'simple':
        return lambda X, Y: 1 - torch.sqrt(torch.mean((X - Y) ** 2)).detach()
    elif task == 'advanced':
        return lambda X, Y: torch.cosine_similarity(X.view(-1), Y.view(-1), dim=0).detach()
    elif task == 'hard':
        net = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
        net.fc = torch.nn.Identity()
        net.eval() 
        return lambda X, Y: torch.cosine_similarity(
            net(X.unsqueeze(0))[0], net(Y.unsqueeze(0))[0], dim=0
        ).detach()
    else:
        raise ValueError("Άγνωστο task. Επιλογές: 'simple', 'advanced', 'hard'")

def main():
    SZ = 128
    TARGET_URL = "https://pdtn.gr/assets/einstein-CY3YjUQr.png"
    TASK_LEVEL = 'simple'
    CUTOFF_SCORE = 0.95

    print(f"Loading target image from {TARGET_URL}...")
    Y = load_target_image(TARGET_URL, SZ)
    
    print(f"Initializing similarity function for task: {TASK_LEVEL}...")
    similarity_fn = get_similarity_function(TASK_LEVEL)

    calls = [0] 

    def score(X, cutoff=CUTOFF_SCORE):
        calls[0] += 1
        s = similarity_fn(X, Y)
        if s > cutoff:
            print(f"Succeeded in {calls[0]} calls!")
            save_image(X, "best_result.png")
            print("Result saved as 'best_result.png'. Exiting...")
            sys.exit(0)
        return s.item()

    X_init = torch.zeros(3, SZ, SZ)
    
    print("Initializing ZOO-ADAM Optimizer...")
    optimizer = ZOOAdam(x_init=X_init, score_fn=score, lr=0.1, box=1, h=0.001)

    print("Starting optimization loop...")
    try:
        while True:
            optimizer.BlockOptimization()
            optimizer.increase()
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user. Saving current best result...")
        save_image(optimizer.get_X(), "interrupted_result.png")

if __name__ == "__main__":
    main()