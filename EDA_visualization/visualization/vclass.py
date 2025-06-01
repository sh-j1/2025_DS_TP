import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os
def EDA_relavant_statics(data_path: str, output_dir: str):
    df = pd.read_csv(data_path)
    os.makedirs(output_dir,exist_ok=True)
    output_path = os.path.join(output_dir,"feature_correlation.png")
    psycho = ['anxiety_level','self_esteem','mental_health_history','depression']
    physical = ['headache','blood_pressure','sleep_quality','breathing_problem']
    environment = ['noise_level','living_conditions','safety','basic_needs']
    academic = ['academic_performance','study_load','teacher_student_relationship','future_career_concerns']
    social = ['social_support','peer_pressure','extracurricular_activities','bullying']
    target = 'stress_level'
    #기본적으로 각 부문에서 stress_level에 어떤 상관관계가 있는지 보여줌
    corr_psycho = df[psycho+[target]].corr()[target].drop(target)
    corr_pyhsical = df[physical+[target]].corr()[target].drop(target)
    corr_environment = df[environment+[target]].corr()[target].drop(target)
    corr_academic = df[academic+[target]].corr()[target].drop(target)
    corr_social = df[social+[target]].corr()[target].drop(target)
    df_psycho = pd.DataFrame({
        'feature':corr_psycho.index,
        'corr':corr_psycho.values,
        'department':'psycho'
    })
    df_pyhsical = pd.DataFrame({
        'feature':corr_pyhsical.index,
        'corr':corr_pyhsical.values,
        'department':'pyhsical'
    })
    df_environment = pd.DataFrame({
        'feature':corr_environment.index,
        'corr':corr_environment.values,
        'department':'environment'
    })
    df_academic = pd.DataFrame({
        'feature':corr_academic.index,
        'corr':corr_academic.values,
        'department':'academic'
    })
    df_social = pd.DataFrame({
        'feature':corr_social.index,
        'corr':corr_social.values,
        'department':'social'
    })
    df_cor=pd.concat([df_psycho,df_pyhsical,df_environment,df_academic,df_social])
    print(df_cor)
    plt.figure(figsize=(10,6))
    sns.barplot(data=df_cor, x='corr', y='feature',hue='department',orient='h')
    plt.tight_layout()
    plt.savefig(output_path)
    #위 상관관계 그래프를 기반으로 각 부문에서 극도로 부정적인 영향에 있는 학생수 비교 - 값이 클수록 학생들에게 취약해지기 쉬운 feature
    output_path = os.path.join(output_dir,"Num_Negative_factor.png")
    num_student_undermean_psycho = (df[psycho] > 0 ).all(axis=1).sum() # 상관관계가 양수여서 수치가 높으면 부정적인 상황에 높인것이라고 판단
    num_student_undermean_physical = (df[physical] > 0).all(axis=1).sum()
    num_student_undermean_environment = (df[environment] < 0).all(axis=1).sum() # 상관관계가 음수여셔 수치가 낮을 수록 부정적인 상황에 높인것으로 판단
    num_student_undermean_academic = (df[academic] < 0).all(axis=1).sum()
    num_student_undermean_social = (df[social] > 0).all(axis=1).sum()

    Num_negative = [num_student_undermean_psycho,num_student_undermean_physical,num_student_undermean_environment,num_student_undermean_academic,num_student_undermean_social]
    x_label = ['psycho','physical','environment','academic','social']
    df_p1 = pd.DataFrame({
        'Factor': x_label,
        'Num_student' : Num_negative
    })
    plt.figure(figsize=(10,6))
    sns.barplot(data=df_p1, x='Factor', y='Num_student')
    plt.title("Number of Students Negative Factor")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    #취약해지기 쉬운 부문 중에 어떤 key feature를 찾아 -> stress level에 큰영향을 주는 feature로 추측 EDA과정 나중에 실제 model 적용 이후와 비교할 예정
    output_path = os.path.join(output_dir,"Environment to Stress.png")
    num_f = len(environment)
    n_col = 2
    n_row = (num_f + n_col-1) // n_col
    plt.figure(figsize=(5*n_col,4*n_row))
    for idx,feature in enumerate(environment,1):
        plt.subplot(n_row,n_col,idx)
        sns.scatterplot(data=df, x=feature, y=target)
        sns.regplot(data=df , x=feature, y=target, scatter=False, color='red')
        plt.xlabel(feature)
        plt.ylabel(target)
    plt.tight_layout()
    plt.savefig(output_path)
    #환경 부문에서는 noise feature가 양의 상관관계를 가지며 극심한 기울기 추세선을 보이는것 safety
    output_path = os.path.join(output_dir,"Social to Stress.png")
    plt.figure(figsize=(5*n_col,4*n_row))
    for idx,feature in enumerate(social,1):
        plt.subplot(n_row,n_col,idx)
        sns.scatterplot(data=df, x=feature, y=target)
        sns.regplot(data=df , x=feature, y=target, scatter=False, color='red')
        plt.xlabel(feature)
        plt.ylabel(target)
    plt.tight_layout()
    plt.savefig(output_path)
    #사회 부문에서는 bullying,peer_pressure가 양의 상관관계를 가졌고 극심한 기울기 추세선을 보이는건 peer_pressure

def visualize_compare_variable_growingstress(data_path: str, output_dir: str, variables: list):
    df = pd.read_csv(data_path)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "growing_stress_comparison.png")
    fig, axs = plt.subplots(nrows=len(variables), ncols=1, figsize=(8, 4 * len(variables)))
    if len(variables)==1:
        axs=[axs]
    for i, var in enumerate(variables):
        sns.boxplot(x='growing_stress', y =var, data=df, ax=axs[i], width=0.5, fliersize=3, boxprops=dict(alpha=0.9), linewidth=1.2)
        axs[i].set_title(f'{var}by growing_stress')
        axs[i].set_xlabel('growing_stress')
        axs[i].set_ylabel(var)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"[Ok] Plot saved to {output_path}")
    plt.close()
