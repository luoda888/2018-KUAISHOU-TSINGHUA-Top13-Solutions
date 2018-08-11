# 2018-KUAISHOU-TSINGHUA-Top13-Solutions
2018中国高校计算机大赛--大数据挑战赛 Top 13 Solutions

#### 初赛A Top 2,初赛B Top 5
	 复赛Final 13 
	 每一次相遇都是久别重逢，下一次站在答辩台上又是何年何月，对得起自己，对得起青春。

#### 题意简述:给出用户在快手APP上1-30日的历史行为，预测接下来7天(31-37)的活跃用户。定义活跃为任意一张表出现过，不考虑冷启动。

##### 框架思考
	1. 等长滑动  [1,16] [2,17] ... [8,23]  ---> Predict [15,30]
	2. 不等长滑动 [1,16] [1,17] ... [1,24] ---> Predict [1,30]

##### 特征工程:
	Tips: 以下Feature都是较为通用的，但是在不等长框架2中，需要对所有涉及到距离的计算做平滑，即除以时间窗口的长度 

	对APP-LAUNCH、ACTION、VIDEO表
		1.对时间序列进行编码
			编码方式考虑两种:
				a. 将用户活跃天数视为二进制数，按二进制方式对活跃天数进行加权，越接近预测日期权重越高，如6天的窗口,1,3,5 --- 101010 将其倒置，010101，转化为十进制数21 
				`ans += binary_day[i]*(2**i)`
				b. 直接按离预测日期距离进行加权
				`ans += binary_day[i]*(1/(end_date-i))`

		2.对时间序列进行描述性统计
			一级统计特征: mean,std,median,max,min,(max-min),mode,count,nunique (在APP表中，count==nunique)
			二级统计特征: skew,kurt,mad
			用户登录的频域周期性: Var(fft(X['day']))
			用户登录的星期周期性: Get_Mode(X['week']) 

		3.对时间序列与预测日期进行时间天数交互
			该用户最后一次登录距离预测日期的长度   end_date+1-x['day'].max()
			该用户倒数第K次登录距离预测日期的长度  end_date+1-get_second_day(x,k)
			最后一次和倒数第二次的距离 x['day'].max()-get_second_day(x,2)

		4.差分一阶时间序列
			用户最大/最小间隔多少天登录一次 Diff Max/Min
			用户平均登录间隔 Diff Mean
			用户登录间隔的稳定性 Diff Var
			用户登录间隔的周期性 Var(fft(X['day'])) 

	对ACTION表的特殊处理
		1.对时间序列进行衰减系数编码
			操作数/当前天数-预测日期的距离 sigma_ans += np.log(ans[i]/(window_len-i))

		2.对时间序列与预测日期进行操作数交互
			该用户最后一次/倒数第二次登录 观看Page、Action的分布 如看了3次Page 0，4次Page 1，2次Action 1，没有的行为用0填充
			将多组行为拆成子图，即可实现统计User在不同页面的分布，如只在Page0发生的行为的描述性统计，mean,std等 (对Action)
			该用户最后一次/K次操作当天，Count VIDEOID/AuthorID
			最后一次和倒数第二次间隔中，Count VIDEOID/AuthorID

		3.对Page，Action进行User的全局展开
			统计每一种行为在该用户整个行为序列里的比例，如 5次点击Page 0，该用户共点击10次 Page 0，那么此处Ratio1 为0.5
			统计每一种行为在当天用户里所有行为的比例，如 该用户900次点击 Page 0，当天共有1000次点击，那么此处Ratio2 为0.9
			上述两种统计，可以有效防止刷单行为，正常的行为序列应当是较为平滑的点击序列，一旦出现峰值，如举报行为居多的，观看同一视频的，即可判定为刷单的特征

		4.计算用户观看VIDEO的贡献序列
			```python
				def GongXianDu(df):
				    d11 = df.set_index('video_id')
				    d11['gongxian_rate'] = df.groupby('video_id').size() 
				    d11['gongxian_rate'] = d11['gongxian_rate'] / d11['video_watched_times']
				    meand = d11['gongxian_rate'].mean()
				    sumd = d11['gongxian_rate'].sum()
				    stdd = d11['gongxian_rate'].std()
				    skeww = d11['gongxian_rate'].skew()
				    kurtt = d11['gongxian_rate'].kurt()
				return sumd,meand,stdd,skeww,kurtt
				temp = train_act.groupby('user_id').apply(GongXianDu)
	    		```

	    5.计算用户是否有追星行为
	    	最喜欢的作者有无更新行为
	    	用户看过的视频还有多少人爱看(区分小众与大众)
	    	```python
		    	def FavAuthorCreate(df):
			    most_author = df.groupby('author_id').size().sort_values(ascending=False).index[0]
			    create_video_num = len(df[df['author_id']==most_author]['video_id'].unique())
			    watch_other_video_num = len(df[df['author_id']==most_author]['video_id'].unique())
			    watch_other_video = 1 if watch_other_video_num>1 else 0
			    return create_video_num , watch_other_video
		```

	对REGSITER表的挖掘
		1.注册周期性，如周末的促销活动，最直观的Feature就是Week
		2.周期性的交互，如在周末特定的Type组合
			register['week'] * register['device_type']
			register['week'] * register['register_type']
		3.类别特征间的交互，如 register['device_type'] * register['register_type']
		4.计算不同类别的使用人数，如 register_log.groupby(['register_type'])['user_id'].transform('count').values
			可以计算device_type，week_rt，week_dt，rt_dt
		5.计算不同类别的转化率(需滑窗计算)，groupby(['count_label_ratio'])['regsiter_type','device_type'].transform('count').vaules

##### 模型选择
	Snake 的 糖尿病特征选择框架 在保留必选特征后(Encoder/FFT) 设置阈值产生两套特征
	树模型: LGB 框架1 
		   XGB 框架2
	FFM : Xlearn 按特征重要性筛选TopK个特征后计算
	xDeepFM 输入序列特征/Category特征
	CNN/RNN 输入序列特征

##### 模型融合
	本题模型融合收益极高，我们尝试了3种方式进行模型融合，按效果来计算
		Top 3 Stacking 没卵用，如果用同一套特征甚至掉分
		Top 2 加权融合 相似性都是0.98 0.97左右，0.97可以有1个千的收益，0.98大概是7-8个万
		Top 1 对半Blending 如用第一个滑窗测试，第二个滑窗测试，两折的Blending 收益是1.2个千








