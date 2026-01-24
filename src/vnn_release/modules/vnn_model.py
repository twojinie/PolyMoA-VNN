# vnn_model.py
import torch.nn as nn, torch
import math

class LinearMasked(nn.Linear):
    """weight ⊙ mask (mask=0 은 완전 고정)."""
    def __init__(self, in_f: int, out_f: int, mask: torch.Tensor):
        super().__init__(in_f, out_f, bias=True)
        self.register_buffer('mask', mask.float())
        self._reset_with_mask()

    # -------- 커스텀 초기화 --------
    def _reset_with_mask(self):
        active = self.mask.sum().item()      # 1의 개수
        if active == 0:
            nn.init.zeros_(self.weight)
        else:
            fan_in = active / self.out_features
            bound  = 1. / math.sqrt(fan_in)
            nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        return nn.functional.linear(x, self.weight * self.mask, self.bias)


class VisibleNN(nn.Module):
    """
    masks: [gene→P5, P5→P4, …, P1→output]  (len = 6)
    마지막 mask는 (1, |P1|) – fully-connected여도 1로 채워서 넘겨 주면 됨.
    """
    def __init__(self, masks):
        super().__init__()
        layers = []
        for m in masks[:-1]:                # gene→P5 … P2→P1 까지 ReLU
            layers += [LinearMasked(m.shape[1], m.shape[0], m), nn.ReLU()]
        # P1→output (binary logit)
        last = masks[-1]
        layers += [LinearMasked(last.shape[1], last.shape[0], last)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):      # x: [B, N_gene]
        logit = self.net(x).squeeze(-1)   # [B]
        return logit             # BCEWithLogitsLoss 에 바로 투입

class VisibleNNWithSkip(VisibleNN):
    def __init__(self, masks, skip_mask):
        super().__init__(masks)

        in_f = skip_mask.shape[1]
        self.skip2out = LinearMasked(in_f, 1, skip_mask)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # skip gate # learnable gate (sigmoid)

    def forward(self, x):
        skips, activs = [], []
        out = x

        # ① self.net 순차 실행
        for layer in self.net:
            out = layer(out)
            if isinstance(layer, LinearMasked):
                activs.append(out)      # 모든 LinearMasked 직후 결과 저장

        # activs 구성
        #  [0] gene→P5  [1] P5→P4  [2] P4→P3  [3] P3→P2
        #  [4] P2→P1    [5] P1→output  ← 마지막은 이미 logit
        logit_base = activs[-1]         # shape [B,1]
        top        = activs[-2]         # P1 activation
        skips      = activs[:-2]        # P5,P4,P3,P2 총 4개
        skip_in    = torch.cat(skips, dim=1)   # shape [B, 3703]
        skip_out = self.skip2out(skip_in)
        gate = torch.sigmoid(self.alpha) # new

        # ② 최종 logit = 기본 + skip 경로
        logit = logit_base + gate * skip_out
        return logit.squeeze(-1)
    
# =====================================================
# 🔹 VNN with skip + gate regularization (λ * α²)
# =====================================================
class VisibleNNWithSkipReg(VisibleNNWithSkip):
    def __init__(self, masks, skip_mask, reg_lambda=1e-3):
        super().__init__(masks, skip_mask)
        self.reg_lambda = reg_lambda

    def skip_reg_loss(self):
        """skip 게이트 α 규제항 (λ * α²)"""
        gate = torch.sigmoid(self.alpha)
        return self.reg_lambda * (gate ** 2)

    def forward(self, x):
        # 그대로 VisibleNNWithSkip forward 사용
        return super().forward(x)