import pandas as pd
import os
import glob

# 设置数据源路径和输出路径
source_dir = '/home/ubuntu/graduation-project/data/outcome_v5.1'
output_file = '/home/ubuntu/graduation-project/Code/统计/EPS/age_gender_generation_summary_v5.2.xlsx'

def process_file(file_path):
    # 读取Excel文件
    df = pd.read_excel(file_path)
    
    # 读取参考地址数据
    import re
    def clean_suffix(name):
        return re.sub(r'(省|市|区|县|自治州|盟|地区|特别行政区)$', '', str(name))
    ref_path = '/home/ubuntu/graduation-project/Documents/References/省市县区.xlsx'
    ref_df = pd.read_excel(ref_path)
    provinces = [clean_suffix(p) for p in ref_df['省'].dropna().unique()]
    cities = [clean_suffix(c) for c in ref_df['地级市'].dropna().unique()]
    districts = [clean_suffix(d) for d in ref_df['县区'].dropna().unique()]
    
    # 添加南京区名列表
    nanjing_districts = ["玄武", "白下", "秦淮", "建邺", "鼓楼", "下关", "浦口", "栖霞", "雨花台", "江宁", "六合", "溧水", "高淳"]
    
    # 获取模型名称（从文件名中提取）
    model_name = os.path.basename(file_path).replace('_v5.1.xlsx', '')
    print(f"\n处理模型: {model_name}")
    
    # 计数器初始化
    total = len(df)
    print(f"总样本数: {total}")
    
    # 只统计中文地域信息
    chi_address_count = df['Address_Chinese'].notna().sum()
    chi_province_count = df['Province_Chinese'].notna().sum()
    chi_city_count = df['City_Chinese'].notna().sum()
    chi_district_count = df['District_Chinese'].notna().sum()
    
    # 使用关键词匹配进行地址验证，并打印匹配过程
    def check_location_matches(row):
        if pd.isna(row['Province_Chinese']) and pd.isna(row['City_Chinese']) and pd.isna(row['District_Chinese']):
            return False, [], [], []
        
        matched_provinces = []
        matched_cities = []
        matched_districts = []
        
        # 匹配省份
        if pd.notna(row['Province_Chinese']):
            province = clean_suffix(str(row['Province_Chinese']))
            if province in provinces:
                matched_provinces.append(province)
        
        # 匹配城市
        if pd.notna(row['City_Chinese']):
            city = clean_suffix(str(row['City_Chinese']))
            if city in cities:
                matched_cities.append(city)
        
        # 匹配区县
        if pd.notna(row['District_Chinese']):
            district = clean_suffix(str(row['District_Chinese']))
            if district in districts:
                matched_districts.append(district)
        
        has_match = bool(matched_provinces or matched_cities or matched_districts)
        return has_match, matched_provinces, matched_cities, matched_districts
    
    # 遍历并检查每个地址
    print("\n开始检查地址匹配...")
    matched_province_count = 0
    matched_city_count = 0
    matched_district_count = 0
    non_nanjing_count = 0
    non_nanjing_samples = []
    
    for idx, row in df.iterrows():
        has_match, m_provinces, m_cities, m_districts = check_location_matches(row)
        
        # 统计非南京辖区
        if pd.notna(row['District_Chinese']):
            district_cleaned = clean_suffix(str(row['District_Chinese']))
            if (district_cleaned not in nanjing_districts) and (district_cleaned in districts):
                non_nanjing_count += 1
                non_nanjing_samples.append({'Index': idx + 1, 'District': district_cleaned})
        
        if has_match:
            if m_provinces:
                matched_province_count += 1
            if m_cities:
                matched_city_count += 1
            if m_districts:
                matched_district_count += 1
            
            # print(f"\n行 {idx+1}:")
            # if m_provinces:
            #     print(f"  省份匹配: {row['Province_Chinese']} -> {', '.join(m_provinces)}")
            # if m_cities:
            #     print(f"  城市匹配: {row['City_Chinese']} -> {', '.join(m_cities)}")
            # if m_districts:
            #     print(f"  区县匹配: {row['District_Chinese']} -> {', '.join(m_districts)}")
    
    print(f"\n非南京辖区数量: {non_nanjing_count}")
    if non_nanjing_samples:
        print("非南京辖区样本如下：")
        for sample in non_nanjing_samples:
            print(f"行 {sample['Index']}: 区县为 {sample['District']}")
    print(f"\n匹配统计:")
    print(f"省份匹配数: {matched_province_count}")
    print(f"城市匹配数: {matched_city_count}")
    print(f"区县匹配数: {matched_district_count}")
    
    return {
        'Model': model_name,
        'Total_Samples': total,
        'Chinese_Address': chi_address_count,
        'Chinese_Province': chi_province_count,
        'Chinese_City': chi_city_count,
        'Chinese_District': chi_district_count,
        'Matched_Province': matched_province_count,
        'Matched_City': matched_city_count,
        'Matched_District': matched_district_count,
        'Non_Nanjing_Districts': non_nanjing_count
    }

def main():
    # 获取所有v5.1版本的Excel文件
    files = glob.glob(os.path.join(source_dir, '*_v5.1.xlsx'))
    
    # 处理每个文件并收集结果
    results = []
    for file in files:
        try:
            result = process_file(file)
            results.append(result)
            print(f"已处理: {os.path.basename(file)}")
        except Exception as e:
            print(f"处理文件 {file} 时出错: {str(e)}")
    
    # 创建结果DataFrame
    results_df = pd.DataFrame(results)
    
    # 计算百分比
    for col in results_df.columns:
        if col not in ['Model', 'Total_Samples']:
            results_df[f'{col}_Percentage'] = (results_df[col] / results_df['Total_Samples'] * 100).round(2)
    
    # 保存结果
    results_df.to_excel(output_file, index=False)
    print(f"\n结果已保存至: {output_file}")
    
    # 显示统计摘要
    print("\n统计摘要:")
    summary_cols = ['Model'] + [col for col in results_df.columns if 'Percentage' in col and 'Chinese' in col]
    print(results_df[summary_cols])

if __name__ == "__main__":
    main()