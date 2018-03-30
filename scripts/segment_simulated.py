from pathlib import Path

from ipmi import template
from ipmi import UnsegmentedSubject
from ipmi import registration as reg

t1_path = Path('/Users/fernando/Desktop/MPHYGB06_coursework_2018/img_99_age_99.nii.gz')

flo_path = template.get_final_template().template_image_path

subject_id = '99'
subject = UnsegmentedSubject(subject_id, t1_age_path=t1_path)


aff_path = subject.template_to_t1_affine_path
cpp_path = subject.template_to_t1_affine_ff_path
ref_path = subject.t1_path
# Linear
args = ref_path, flo_path
kwargs = {'trsf_path': aff_path}
reg.register(*args, **kwargs)

res_path = subject.template_on_t1_affine_ff_path
# Free-form
args = ref_path, flo_path
kwargs = {'trsf_path': cpp_path,
          'init_trsf_path': aff_path,
          'res_path': res_path,
         }
reg.register_free_form(*args, **kwargs)


subject.propagate_priors(non_linear=True)


subject.segment()
