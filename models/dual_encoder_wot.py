import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class RankModel(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.encoder_type = args.encoder_type
        if self.encoder_type == 'biobert':
            self.encoder = BertModel.from_pretrained(args.plm_path)
        self.pooler_type = args.pooler_type
        self.matching_func = args.matching_func
        self.temperature = args.temperature if hasattr(args, 'temperature') else 1
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.softmax = nn.Softmax(dim=1)
    
    def pooler(self, last_hidden_state, attention_mask):
        if self.pooler_type == 'cls':
            pooler_output = last_hidden_state[:, 0]
        if self.pooler_type == 'mean':
            pooler_output = torch.sum(last_hidden_state * attention_mask.unsqueeze(2), dim=1)
            pooler_output /= torch.sum(attention_mask, dim=1, keepdim=True)
        return pooler_output

    def sentence_encoding(self, input_ids, attention_mask):
        if self.encoder_type == 'biobert':
            encoder_outputs = self.encoder(input_ids=input_ids, 
                                            attention_mask=attention_mask,
                                            output_hidden_states=True,
                                            return_dict=True)
            last_hidden_state = encoder_outputs['last_hidden_state']

        sentence_embeddings = self.pooler(last_hidden_state, attention_mask)
        
        return sentence_embeddings
    
    def matching(self, src_embeddings, tgt_embeddings, src_ids=None, tau_scale=True):

        if self.matching_func == "cos":
            src_embeddings = F.normalize(src_embeddings, dim=-1)
            tgt_embeddings = F.normalize(tgt_embeddings, dim=-1)
        predict_logits = src_embeddings.mm(tgt_embeddings.t())
        if tau_scale:
            predict_logits /= self.temperature

        if src_ids is not None:
            batch_size = src_embeddings.shape[0]
            logit_mask = (src_ids.unsqueeze(1).repeat(1, batch_size) == src_ids.unsqueeze(0).repeat(batch_size, 1)).float() - torch.eye(batch_size).to(src_ids.device)
            predict_logits -= logit_mask * 100000000

        return predict_logits

    def forward(self, src_input_ids, src_attention_mask, tgt_input_ids, tgt_attention_mask, src_ids=None):
        # obtain sentence embeddings
        src_embeddings = self.sentence_encoding(src_input_ids, src_attention_mask)
        tgt_embeddings = self.sentence_encoding(tgt_input_ids, tgt_attention_mask)
 
        if self.training:
            # matching matrix with shape of [bs, bs]
            predict_logits = self.matching(src_embeddings, tgt_embeddings, src_ids, True)
            # reqa loss
            labels = torch.arange(0, predict_logits.shape[0]).to(src_input_ids.device)
            predict_loss = self.ce_loss(predict_logits, labels)
            # wot loss
            sentence_embeddings = torch.cat([src_embeddings, tgt_embeddings], 0)
            mean_vector = torch.mean(sentence_embeddings, dim=0, keepdims=True)
            cov = (sentence_embeddings-mean_vector).t().mm((sentence_embeddings-mean_vector)) / (sentence_embeddings.shape[0]-1)
            cov_label = torch.eye(sentence_embeddings.shape[1]).to(sentence_embeddings.device)
            predict_loss += self.mse_loss(cov, cov_label)
            # accuracy
            predict_result = torch.argmax(predict_logits, dim=1)
            acc = labels == predict_result
            acc = (acc.int().sum() / (predict_logits.shape[0] * 1.0)).item()

            return predict_loss, acc
        else:
            return src_embeddings, tgt_embeddings, src_ids














# def pooler(self, encoder_outputs, attention_mask):
#         if isinstance(encoder_outputs, dict):
#             hidden_states = encoder_outputs['hidden_states']
#         else:
#             hidden_states = encoder_outputs[2]
        
#         first_hidden_state = hidden_states[1]
#         last_hidden_state = hidden_states[-1]

#         if self.pooler_type == 'cls':
#             pooler_output = last_hidden_state[:, 0]
#         if self.pooler_type == 'clsmlp':
#             pooler_output = self.cls_mlp(last_hidden_state[:, 0])
#         if self.pooler_type == 'mean':
#             pooler_output = torch.sum(last_hidden_state * attention_mask.unsqueeze(2), dim=1)
#             pooler_output /= torch.sum(attention_mask, dim=1, keepdim=True)
#         if self.pooler_type == 'flmean':
#             pooler_output = torch.sum(first_hidden_state * attention_mask.unsqueeze(2), dim=1)
#             pooler_output += torch.sum(last_hidden_state * attention_mask.unsqueeze(2), dim=1)
#             pooler_output /= torch.sum(attention_mask, dim=1, keepdim=True) * 2
#         if self.pooler_type == 'att':
#             att_logits = self.att_layer(last_hidden_state) - 1000 * (1-attention_mask.float().cuda()).unsqueeze(2)
#             att_scores = self.softmax(att_logits).transpose(2, 1)
#             pooler_output =  torch.bmm(att_scores, last_hidden_state).squeeze(1)
#         if self.pooler_type == 'flatt':
#             first_att_logits = self.att_layer(first_hidden_state) - 1000 * (1-attention_mask.float().cuda()).unsqueeze(2)
#             last_att_logits = self.att_layer(last_hidden_state) - 1000 * (1-attention_mask.float().cuda()).unsqueeze(2)
#             # [bs, 2*len, 1]
#             att_scores = self.softmax(torch.cat([first_att_logits, last_att_logits], 1))
#             seq_len = int(att_scores.shape[1] / 2)
#             # # [bs, 1, len]
#             first_att_scores = att_scores[:, :seq_len].transpose(2, 1)
#             last_att_scores = att_scores[:, seq_len:].transpose(2, 1)

#             pooler_output =  torch.bmm(first_att_scores, first_hidden_state).squeeze(1)
#             pooler_output +=  torch.bmm(last_att_scores, last_hidden_state).squeeze(1)
        
#         return pooler_output