from .formating import *
from .recognition_label_convert import CTCLabelConverter, CTCLabelConverterForBaiduWarpctc, AttnLabelConverter


@PIPELINES.register_module
class TextDefaultFormatBundle(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                       (3)to DataContainer (stack=True)
    """
    def __init__(self, character=[], recognition_predition_type='Attn'):
        # self.character = character
        self.character = character
        if recognition_predition_type == 'Attn':
            self.converter = AttnLabelConverter(self.character)
        elif recognition_predition_type == 'CTC':
            self.converter = CTCLabelConverter(self.character)
        elif recognition_predition_type == 'BaiduCTC':
            self.converter = CTCLabelConverterForBaiduWarpctc(self.character)
        else:
            raise ValueError('recognition predition type should in [Attn, CTC, BaiduCTC]')

    def __call__(self, results):
        if 'gt_transcriptions' in results:
            results['gt_transcriptions'], results['gt_decoders'], results['gt_text_lengths'] = self.converter.encode(results['gt_transcriptions'])
            results['poly_fields'].extend(['gt_text_lengths', 'gt_decoders'])

        if 'img' in results:
            img = np.ascontiguousarray(results['img'].transpose(2, 0, 1))
            results['img'] = DC(to_tensor(img), stack=True)
        for key in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]))
        if 'gt_masks' in results:
            results['gt_masks'] = DC(results['gt_masks'], cpu_only=True)

        for key in results['poly_fields']:
            if key == 'gt_ori_poly':
                continue
            results[key] = DC(to_tensor(results[key]))
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None, ...]), stack=True)
        return results

    def __repr__(self):
        return self.__class__.__name__
