#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np
import joblib
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the page configuration
st.set_page_config(page_title='Maternal Health Care Guide', layout='wide')
st.markdown('<div class="header">Welcome to the Maternal Health Care Guide</div>', unsafe_allow_html=True)

# 自定义 CSS 样式
st.markdown("""
<style>
img {
    width: 100%;  /* Adjust based on column width in Streamlit */
    height: auto;
    margin-top: 10px;
}
.header {
    background-color: #f8f8f8;
    padding: 10px 20px;
    text-align: center;
    font-size: 44px;
    font-weight: bold;
}
.button {
    
    display: inline-block;
    padding: 0.5em 2em;
    margin: 0 0.3em 0.3em 0;
    border-radius: 0.15em;
    box-sizing: border-box;
    text-decoration: none;
    font-family: 'Roboto',sans-serif;
    font-weight: 300;
    color: #FFFFFF;
    background-color: #4CAF50;
    text-align: center;
    transition: all 0.2s;
}
.button:hover {
    background-color: #45a049;
}
.stButton>button {
    background-color: #f8f8f8;
    display: block;
    width: 100%;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# Use columns to create a navigation menu at the top of the page
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.button('Home'):
        st.session_state.page = 'Home'
with col2:
    if st.button('Depression Prediction'):
        st.session_state.page = 'Depression Prediction'
with col3:
    if st.button('Recipes '):
        st.session_state.page = 'Recipes'

# Set the default page if not already set
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# Home page content
if st.session_state.page == 'Home':
    
    st.markdown("""
        <style>
        img {
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
        </style>
        """, unsafe_allow_html=True)
    image_path = r'C:\Users\ACER\Desktop\研究生\第四学期\论文\banner.png'  # Update the path to your image
    
    # Display the image with specified width adjustments
    st.image(image_path, caption=None, use_column_width=True)

    # Overlay title on the image using custom CSS
    st.markdown("""
        <style>
        .overlay-text {
            font-size:50px;
            font-weight: bold;
            color: white;
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
            width: 100%;
        }
        </style>
        <div class="overlay-text">Welcome to the Maternal Health Care Guide</div>
        """, unsafe_allow_html=True)

    st.header('Maternity care: Healthy pregnancy, making expectant mothers happy and babies healthier.')
    st.write("""
    Prenatal care is an essential part of a healthy pregnancy. Whether you are planning to get pregnant soon, or if you just found out that you are expecting, it’s important to focus on your prenatal care. Good prenatal care can reduce the risk of complications, ensure a smoother delivery, and help ensure both you and your baby are healthy throughout the pregnancy.
 In daily life, in addition to caring about the physical health of pregnant women, we should also pay attention to their psychological and emotional states. Negative emotions such as anxiety and irritability in pregnant women can also affect the baby's health. So, what impact do negative emotions in pregnant women have on the baby? And what should pregnant women do in the face of various negative emotions that may arise during pregnancy?
    
    Negative emotions in pregnant women can lead to significant problems for the baby. According to a report by the "People's Daily Overseas Edition," excessive worry and tension in pregnant women can cause loss of appetite and indigestion, which affects the absorption of nutrients; tension can also cause vasoconstriction and increased blood pressure, significantly reducing blood supply to the fetus, leading to delayed fetal development, etc. Therefore, it can be said that the mood of a pregnant woman is like the "spiritual nourishment" for the fetus. Why does a mother's poor mood during pregnancy affect the fetus, even though there is no direct contact between the nervous systems of the mother and the fetus? This is because emotional stimulation in the mother can activate the autonomic nervous system, leading to the release of acetylcholine and changes in the type and amount of hormone secretion. These substances enter the fetus through the blood, placenta, and umbilical cord, thereby affecting its physical and mental health.
    
    How should pregnant women handle various negative emotions that may arise during pregnancy? According to "Health Daily," the following suggestions are offered:
    
    - Pregnant mothers should learn about childbirth to avoid excessive fear of it, recognizing that childbirth is an inevitable outcome of pregnancy.
    - Pregnant mothers can prepare some essentials for the unborn child with their family, which can change the pregnant woman's fear of childbirth to eager anticipation, helping to improve her mood.
    - Pregnant women in their mid-term can do some work and participate in some gentle exercises. Unless there are exceptional circumstances, they should continue to work normally, as this also benefits their psychological state during pregnancy.
    """)

elif st.session_state.page == 'Depression Prediction':
     # 载入你的模型和scaler 
    scaler_path = r'C:\Users\ACER\Desktop\研究生\第四学期\论文\scaler.pkl'
    model_path = r'C:\Users\ACER\Desktop\研究生\第四学期\论文\lr_classifier_model.pkl'
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)

    # 设置Streamlit页面配置
    
    st.title('Prediction of prenatal depression risk')

    # 创建表单以接受用户输入
    with st.form('prediction_form'):
        st.header('Please enter the following information:')

        Social_Support = st.number_input('Social Support(0-100)', min_value=1, max_value=100, value=50)
        background = st.number_input('background(0-Unemployed,1-Entry Level,2-Mid Level,3-Senior Level)', min_value=0, max_value=3, value=1)
        Used_Planning = st.number_input('Used Planning(1-Occasional Use,2-Regular Use,3-Continuous Use)', min_value=1, max_value=3, value=1)
        Household_decision = st.number_input('Household Decision(1-External Decision,2-Family Joint Decision,3-Partner Joint Decision,4-Individual Decision)', min_value=1, max_value=4, value=1)
        Education = st.number_input('Education(0-Illiterate,1-Primary School,2-Middle School,3-High School,4-Undergraduate,5-Master,6-PhD)', min_value=0, max_value=6, value=3)
        Household_Income = st.number_input('Household Income(1: 0-20,000, 2: 20,000-50,000, 3: 50,000-100,000, 4: 100,000+)', min_value=1, max_value=4, value=1)
        Anxiety = st.number_input('Anxiety(0-20)', min_value=0, max_value=20, value=5)
        Pregnancy = st.number_input('Pregnancy', min_value=1, max_value=2, value=1)
        Duration_Marriage = st.number_input('Duration of Marriage', min_value=0, max_value=100, value=5)
        Age_Maternalnew = st.number_input('Age Maternal', min_value=0, max_value=100, value=30)

        # 添加提交按钮
        submit_button = st.form_submit_button(label='Predict')

    # 处理表单提交
    if submit_button:
        features = np.array([[Social_Support, background, Used_Planning, Household_decision, Education,
                              Household_Income, Anxiety, Pregnancy, Duration_Marriage, Age_Maternalnew]])

        # 特征标准化
        features_scaled = scaler.transform(features)

        # 使用模型进行预测
        prediction = model.predict(features_scaled)
        prediction_proba = model.predict_proba(features_scaled)

        # 显示预测结果
        st.subheader('Predicting results')
        if prediction[0] == 1:
            st.warning(f'Prediction: High risk of prenatal depression (Probability {prediction_proba[0][1]:.2f})')
            st.info('Please note, your test result is concerning. It is advisable to seek a comprehensive medical evaluation at a healthcare facility. Regular monitoring and possible interventions could be essential for your health.')
        else:
            st.success(f'Prediction: Low risk of prenatal depression (Probability {prediction_proba[0][1]:.2f})')
            st.info('Congratulations, your results are normal. However, it’s important to maintain regular health check-ups and seek medical advice if you experience any symptoms related to depression.')
        # 载入数据
    @st.cache_data  # 使用缓存装饰器提高加载速度
    def load_data():
        file_path = r'C:\Users\ACER\Desktop\研究生\第四学期\论文\data3.csv'
        return pd.read_csv(file_path)

    data = load_data()



# 更新数据并绘制图表
    col1, col2 = st.columns([2, 2])  # 创建两列，图表列宽度更大
    data['Education'] = data['Education'].astype(int)

    with col1:
    # 社会支持与抑郁得分分析
       st.title('Social Support vs. Depression Scores')
       min_support, max_support = int(data['Social_Support'].min()), int(data['Social_Support'].max())
       social_support = st.slider('Select range of Social Support', min_support, max_support, (min_support, max_support))
    # 更新数据并绘制图表
       filtered_data = data[(data['Social_Support'] >= social_support[0]) & (data['Social_Support'] <= social_support[1])]
       fig, ax = plt.subplots(figsize=(8, 4))  # 控制图表尺寸
       sns.scatterplot(x='Social_Support', y='Depression', data=filtered_data, ax=ax)
       plt.title('Social Support vs. Depression Scores')
       st.pyplot(fig)

    with col2:
    # 教育水平与抑郁得分分析
        st.title('Education Level vs. Depression Scores')
       #education_levels = sorted(data['Education'].unique())
    
    # 创建教育水平的映射字典
        education_map = {
            0: 'Illiterate',
            1: 'Primary School',
            2: 'Middle School',
            3: 'High School',
            4: 'Undergraduate',
            5: 'Master',
            6: 'PhD'
        }

        # 更新数据集中的教育水平
        data['Education'] = data['Education'].map(education_map)

        # 获取所有唯一的教育水平并排序
        education_levels = sorted(data['Education'].unique())
        
        # 为教育水平添加滑块选择器
        education_selected = st.multiselect('Select Education Levels', options=education_levels, default=education_levels)
            
        # 根据选择的教育水平过滤数据
        filtered_education_data = data[data['Education'].isin(education_selected)]
        filtered_education_data = filtered_education_data.sort_values(by='Education', key=lambda x: x.map(lambda y: list(education_map.values()).index(y)))
        
        # 绘制条形图
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        sns.barplot(x='Education', y='Depression', data=filtered_education_data, ax=ax2)
        plt.title('Education Level vs. Depression Scores')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        st.pyplot(fig2)

    col3, col4 = st.columns([2, 2])  # 创建两列，图表列宽度相等

    with col3:
    # 家庭收入与抑郁得分分析
        st.title('Household Income vs. Depression Scores')
        # 定义家庭收入的映射字典
        income_map = {
            1: '0-20,000',
            2: '20,000-50,000',
            3: '50,000-100,000',
            4: '100,000+'
        }

        # 更新数据集中的家庭收入标签
        data['Household_Income'] = data['Household_Income'].map(income_map)

        # 获取所有唯一的家庭收入等级并排序
        order = ['0-20,000', '20,000-50,000', '50,000-100,000', '100,000+']

        # 添加多选框以选择家庭收入等级
        income_selected = st.multiselect('Select Household Income Levels', options=order, default=order)
        
        # 根据选择的家庭收入等级过滤数据
        filtered_income_data = data[data['Household_Income'].isin(income_selected)]
        
        # 绘制条形图
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        sns.barplot(x='Household_Income', y='Depression', data=filtered_income_data, order=order, ax=ax3)
        plt.title('Household Income(PKR/M) vs. Depression Scores')
        ax3.set_xlabel('Household Income (PKR)') 
        st.pyplot(fig3)

    with col4:
        # 另一个示例图表
        # 假设这里我们用另一个指标 'Anxiety' 来进行分析
        st.title('Anxiety Scores vs. Depression Scores')
        min_anxiety, max_anxiety = int(data['Anxiety'].min()), int(data['Anxiety'].max())
        anxiety_range = st.slider('Select range of Anxiety', min_anxiety, max_anxiety, (min_anxiety, max_anxiety))
        filtered_anxiety_data = data[(data['Anxiety'] >= anxiety_range[0]) & (data['Anxiety'] <= anxiety_range[1])]
        fig4, ax4 = plt.subplots(figsize=(8, 4))
        sns.scatterplot(x='Anxiety', y='Depression', data=filtered_anxiety_data, ax=ax4)
        plt.title('Anxiety Scores vs. Depression Scores')
        st.pyplot(fig4)   
elif st.session_state.page == "Recipes":
    st.markdown('<h1>Nutrition Recommendations</h1>', unsafe_allow_html=True)
    # Example: Embedding images and descriptions
    #st.image('https://c.ndtvimg.com/2022-03/dt76ej2_omega-3-plant-based-foods_625x300_23_March_22.jpg', caption='Omega-3 Fatty Acids')
    st.markdown("""
    <div style="text-align:left;">
        <img src="https://c.ndtvimg.com/2022-03/dt76ej2_omega-3-plant-based-foods_625x300_23_March_22.jpg" alt="Omega-3 Fatty Acids" style="width:600px;height:300px;">
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    #### 1. Omega-3 Fatty Acids
    
    Omega-3 fatty acids are essential fatty acids that have been found to significantly reduce the risk for sudden death caused by irregular heartbeat and other causes in patients with known heart disease. They are important for heart health and brain health.
    
    The two most crucial omega-3 fatty acids are Eicosapentaenoic Acid (EPA) and Docosahexaenoic Acid (DHA).
    
    **Recommendation:** 2 - 4g per day for people with hypertriglyceridemia
    
    - Walnuts (¼ cup = 2.3g)
    - Chia seeds (1 tbsp = 1.9g)
    - Salmon (75g = 1.9g)
    - Mackerel (75g = 1g)
    - Canola oil (1 tsp = 0.6g)

    #### 2. Fibre and Wholegrain
    """)
    #st.image('https://images.rodpub.com/images/183/599_main.jpg', caption='Fibre and Wholegrain')
    st.markdown("""
    <div style="text-align:left;">
        <img src="https://images.rodpub.com/images/183/599_main.jpg" alt="Fibre and Wholegrain" style="width:500px;height:300px;">
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""

    Fibre promotes gut health and reduces the risk of developing many chronic diseases. The recommended fibre intake is five servings of fruits and vegetables per day.
    
    **Recommendation:** 25 - 30g fibre per day (7 - 13g soluble fibre)
    
    - Vegetables (½ cup = 5g fibre)
    - Fruits (1 serving = 4.4g fibre)
    - Black beans (½ cup = 15g total fibre, 3.6g soluble fibre)
    - Sweet potato (½ cup = 2g total fibre, 1.8g soluble fibre)
    - Avocado (½ fruit, 6.5g total fibre, 2.1g soluble fibre)

    #### 3. Beta Glucan

    """)
    #st.image('https://i0.wp.com/post.healthline.com/wp-content/uploads/2021/11/grains-oats-maize-Beta-Glucan-fiber-1296x728-header.jpg?w=1155&h=1528', caption='Beta Glucan')
    st.markdown("""
    <div style="text-align:left;">
        <img src="https://i0.wp.com/post.healthline.com/wp-content/uploads/2021/11/grains-oats-maize-Beta-Glucan-fiber-1296x728-header.jpg?w=1155&h=1528" alt="Beta Glucan" style="width:500px;height:300px;">
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""

    Beta glucan is a soluble fibre that forms a gel in the digestive tract. It binds to excess cholesterol and cholesterol-like substances within the gut and prevents these from being absorbed by the body.
    
    **Recommendation:** ≥ 3 g reduces LDL cholesterol up to 5%

    - Oatmeal (3 tablespoons, raw = 1g)
    - Barley (1/2 cup cooked = 3.5g)
    - Oat bran powder (2 scoops = 3g)
    - Fortified products (1 serving = 0.8g)

    #### 4. Soy Protein

    """)
    st.markdown("""
    <div style="text-align:left;">
        <img src="https://www.healthkart.com/connect/wp-content/uploads/2022/03/900x500_banner_HK-Soy-protein-benefits.jpg" alt="Soy Protein" style="width:500px;height:300px;">
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    
    Soy protein is found in soybeans and soy products. It contains zero cholesterol, is low in saturated fat, and is the only plant-sourced food that contains all eight essential amino acids.
    
    **Recommendation:** 25g per day reduces LDL cholesterol up to 4 - 5%

    - Firm tofu (1 large block, 350g = 18g)
    - Tempeh (90g = 13g)
    - Soymilk (1 cup = 7g)
    - Edamame (½ cup = 11g)
    #### 5. Nuts

    """)
    st.markdown("""
    <div style="text-align:left;">
        <img src="https://www.allrecipes.com/thmb/Wg9OMnGfdNUmqSVT_HhN9BX9OQY=/750x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/How-to-Store-Nuts-3x2-1-47cd3e4e561c466984fd00251a5d6f70.png" alt="Nuts" style="width:500px;height:300px;">
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    
   Tree nuts and peanuts are rich in unsaturated fatty acids, nutrients, and other bioactive compounds such as protein, fibre, minerals, tocopherols, phytosterols, and phenolic compounds.
    
    **Recommendation:** 150g per week (30g for 5 days per week) lowers cholesterol by an average of 5% and reduces heart disease risk by 37%

    - Almond
    - Almond
    - Cashew nuts
    - Walnuts
    - Pistachio
    - Macadamia
    - Peanuts
    - Pecan
    #### 6. Plant Sterols

    """)
    st.markdown("""
    <div style="text-align:left;">
        <img src="https://blog.insidetracker.com/hs-fs/hubfs/social-suggested-images/shelllfish%20and%20cholesterol.png?width=368&name=shelllfish%20and%20cholesterol.png" alt="Plant Sterols" style="width:500px;height:500px;">
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    Plant sterols, also known as phytosterols, are structurally similar to cholesterol in the human body, thus they compete with cholesterol for absorption in the digestive system.
    
    **Recommendation:**  2 - 3g per day reduces LDL cholesterol up to 10%

    - Chickpeas (1 cup = 0.24g)
    - Corn oil (2 tbsp = 0.3g)
    - Fortified milk (250ml = 0.83g)
  

    """)
# In[ ]:

