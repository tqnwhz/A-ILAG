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
        self.temperature = args.temperature if hasattr(args,
                                                       'temperature') else 1
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
        self.softmax = nn.Softmax(dim=1)
        self.adv_norm = args.adv_norm

    def pooler(self, last_hidden_state, attention_mask):
        if self.pooler_type == 'cls':
            pooler_output = last_hidden_state[:, 0]
        if self.pooler_type == 'mean':
            pooler_output = torch.sum(last_hidden_state *
                                      attention_mask.unsqueeze(2),
                                      dim=1)
            pooler_output /= torch.sum(attention_mask, dim=1, keepdim=True)
        return pooler_output

    def sentence_encoding(self,
                          input_ids,
                          attention_mask,
                          is_query=True,
                          inputs_embeds=None):
        # device = input_ids.device if input_ids!=None else inputs_embeds.device
        # if is_query:
        #     token_type_ids = torch.zeros(input_ids.shape, device=device,dtype=torch.long)
        # else:
        #     token_type_ids = torch.ones(input_ids.shape, device=device,dtype=torch.long)
        if self.encoder_type == 'biobert':
            encoder_outputs = self.encoder(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           output_hidden_states=True,
                                           inputs_embeds=inputs_embeds,
                                           return_dict=True)
            last_hidden_state = encoder_outputs['last_hidden_state']
            sentence_embeddings = self.pooler(last_hidden_state,
                                              attention_mask)

        return sentence_embeddings

    def matching(self,
                 src_embeddings,
                 tgt_embeddings,
                 src_ids=None,
                 tau_scale=True):
        if self.matching_func == "cos":
            src_embeddings = F.normalize(src_embeddings, dim=-1)
            tgt_embeddings = F.normalize(tgt_embeddings, dim=-1)
        predict_logits = src_embeddings.mm(tgt_embeddings.t())
        if tau_scale:
            predict_logits /= self.temperature

        if src_ids is not None:
            batch_size = src_embeddings.shape[0]
            logit_mask = (src_ids.unsqueeze(1).repeat(
                1, batch_size) == src_ids.unsqueeze(0).repeat(
                    batch_size, 1)).float() - torch.eye(batch_size).to(
                        src_ids.device)
            predict_logits -= logit_mask * 100000000

        return predict_logits

    def adv_training(self,
                     src_input_ids,
                     src_attention_mask,
                     tgt_input_ids,
                     tgt_attention_mask,
                     src_ids=None):
        src_inputs_embeds = self.encoder.embeddings.word_embeddings(
            src_input_ids)
        tgt_inputs_embeds = self.encoder.embeddings.word_embeddings(
            tgt_input_ids)
        src_inputs_embeds = src_inputs_embeds.clone().detach()
        src_inputs_embeds.requires_grad = True
        tgt_inputs_embeds = tgt_inputs_embeds.clone().detach()
        tgt_inputs_embeds.requires_grad = True
        loss, *_ = self(None,
                        src_attention_mask,
                        None,
                        tgt_attention_mask,
                        src_ids=src_ids,
                        src_inputs_embeds=src_inputs_embeds,
                        tgt_inputs_embeds=tgt_inputs_embeds)
        src_grad, tgt_grad = torch.autograd.grad(
            loss, [src_inputs_embeds, tgt_inputs_embeds],
            retain_graph=False,
            create_graph=False)

        src_inputs_embeds = src_inputs_embeds + self.adv_norm * src_grad.sign()
        tgt_inputs_embeds = tgt_inputs_embeds + self.adv_norm * tgt_grad.sign()
        return self(None,
                    src_attention_mask,
                    None,
                    tgt_attention_mask,
                    src_ids=src_ids,
                    src_inputs_embeds=src_inputs_embeds,
                    tgt_inputs_embeds=tgt_inputs_embeds)

    def forward(self,
                src_input_ids,
                src_attention_mask,
                tgt_input_ids,
                tgt_attention_mask,
                src_ids=None,
                src_inputs_embeds=None,
                tgt_inputs_embeds=None,
                adv_training=False):
        if adv_training:
            return self.adv_training(src_input_ids,
                                     src_attention_mask,
                                     tgt_input_ids,
                                     tgt_attention_mask,
                                     src_ids=src_ids)
        # obtain sentence embeddings
        src_embeddings = self.sentence_encoding(
            src_input_ids,
            src_attention_mask,
            inputs_embeds=src_inputs_embeds,
            is_query=True)
        tgt_embeddings = self.sentence_encoding(
            tgt_input_ids,
            tgt_attention_mask,
            inputs_embeds=tgt_inputs_embeds,
            is_query=False)

        batch_size = src_embeddings.shape[0]
        if self.training:
            # matching matrix with shape of [bs, bs]
            predict_logits = self.matching(src_embeddings, tgt_embeddings,
                                           src_ids, True)
            # loss
            labels = torch.arange(0, predict_logits.shape[0]).to(
                predict_logits.device)
            predict_loss = self.ce_loss(predict_logits, labels)
            # accuracy
            predict_result = torch.argmax(predict_logits, dim=1)
            acc = labels == predict_result
            acc = (acc.int().sum() / (predict_logits.shape[0] * 1.0)).item()

            predict_distribution = predict_logits.detach().softmax(-1)
            # group_mask = self.group_mask.to(predict_distribution.device)[:batch_size,:batch_size]
            # difficulty = predict_distribution.masked_select(
            #     group_mask).sum() / batch_size
            pair_mask = torch.eye(batch_size, dtype=torch.bool).to(
                predict_distribution.device)
            difficulty = predict_distribution.masked_select(
                ~pair_mask).sum() / batch_size  # / (batch_size - 1)
            return predict_loss, difficulty, acc
        else:
            return src_embeddings, tgt_embeddings, src_ids
