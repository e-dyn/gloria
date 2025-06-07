{{ fullname | escape | underline}}

{% set excluded_attributes = ['model_config', 'model_fields', 'model_extra', 'model_fields_set', 'model_computed_fields'] %}
{% set excluded_members = [
    'Foster',
    'make_all_features',
    'model_config',
    '__init__',
    'preprocess',
    'set_changepoints',
    'time_to_integer',
    'validate_changepoints',
    'validate_column_name',
    'validate_dataframe',
    'validate_metric_column',
    'validate_population_name',
    'validate_sampling_period'
] %}

{% set non_inherited_attributes = attributes 
    | reject('in', inherited_members) 
    | reject('in', excluded_members) 
    | select('in', members) 
    | list %}
    
{% set non_inherited_methods = methods 
    | reject('in', inherited_members) 
    | reject('in', excluded_members) 
    | select('in', members) 
    | list %}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}

   {% block attributes %}
   {% if non_inherited_attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in non_inherited_attributes %}
        ~{{ name }}.{{ item }}
   {% endfor %}
   {% endif %}
   {% endblock %}

   {% if non_inherited_methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in non_inherited_methods %}      
        ~{{ name }}.{{ item }}
   {% endfor %}
   {% endif %}
   {% endblock %}
   