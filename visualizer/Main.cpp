# include <Siv3D.hpp> // Siv3D v0.6.12




String getEpisodeFromString(const String& s) {
	if (not s.contains(U"_")) {
		return s;
	}
	// path で一番最後にある_以降だけ残す
	return s.substr(1 + s.lastIndexOf(U"_"));
}

String getEpisodeNumberFromString(const String& s) {
	String t = getEpisodeFromString(s);
	if (not t.contains(U".")) {
		return U"";
	}
	t = t.substr(0, t.lastIndexOf(U"."));
	return t;// Parse<int32>(t);
}



struct State {
	int32 width, height;
	Array<Vec2> myPos;
	int32 mySize;
	int32 enemyNum;
	Array<Array<Vec2>> enemyPos;
	Array<int32> enemySize;
	bool win = false;
	double totalReward = 0;
	State() :width(0), height(0), myPos(0), mySize(0), enemyNum(0), enemyPos(0), enemySize(0) {
		reset();
	}
	void reset() {
		myPos.clear();
		enemyPos.clear();
		enemySize.clear();
	}
};


struct StatisticsData {
	int32 episodeNum;
	Array<bool> win;
	Array<double> totalReward;
	Array<double> winRate;
	int32 width;
	int32 calc_width;
	StatisticsData() :episodeNum(0), win(0), totalReward(0), winRate(0), width(100), calc_width(0){}
	void reset() {
		episodeNum = 0;
		win.clear();
		totalReward.clear();
		winRate.clear();
		width = 100;
	}
	void calc() {
		// 直近 width 回の成功率を計算する
		int16 winCount = 0;
		winRate.clear();
		for (size_t i : step(episodeNum)) {
			winCount += (win[i] ? 1 : 0);
			if (i >= width) {
				winCount -= (win[i - width] ? 1 : 0);
			}
			winRate.push_back((double)winCount / (double)width);
		}
		calc_width = width;
	}
};

State getStateFromCsv(String path) {
	State state;

	CSV csv{ path };
	if (not csv) {
		throw Error(U"CSVファイルの読み込みに失敗しました");
	}

	state.reset();
	size_t idx = 0;
	state.height = Parse<int32>(csv[0][idx++]);
	state.width = Parse<int32>(csv[0][idx++]);
	state.mySize = (int32)Parse<float>(csv[0][idx++]);
	int32 len = Parse<int32>(csv[0][idx++]);
	for (auto i : step(len)) {
		float x = Parse<float>(csv[0][idx++]);
		float y = Parse<float>(csv[0][idx++]);
		state.myPos << Vec2{ x,y };
	}
	len = Parse<int32>(csv[0][idx++]);
	state.enemyNum = len;
	for (auto i : step(len)) {
		Array<Vec2> pos;
		state.enemySize << (int32)Parse<float>(csv[0][idx++]);
		int32 len2 = Parse<int32>(csv[0][idx++]);
		for (auto j : step(len2)) {
			float x = Parse<float>(csv[0][idx++]);
			float y = Parse<float>(csv[0][idx++]);
			pos << Vec2{ x,y };
		}
		state.enemyPos << pos;
	}
	state.win = (int32)Parse<float>(csv[0][idx++]) == 1;
	// state.win = state.myPos.back().y < 20;
	state.totalReward = Parse<double>(csv[0][idx++]);
	// state.win = RandomBool();



	return state;
}



void Main()
{
	// 背景の色を設定する | Set the background color
	// Scene::SetBackground(ColorF{ 0.6, 0.8, 0.7 });
	Scene::SetBackground(ColorF{ 0.9 });
	ListBoxState listBoxState;
	Window::Resize(1200, 800);
	Scene::SetResizeMode(ResizeMode::Keep);
	Window::SetStyle(WindowStyle::Sizable);
	String path;
	String sedai;
	Array<String> pathList;
	Font hugeFont{ FontMethod::MSDF, 64, Typeface::Bold };
	Font mediumFont{ FontMethod::MSDF, 24, Typeface::Medium };
	Font tinyFont{ FontMethod::MSDF, 12, Typeface::Medium };
	State state;
	int16 stateT = 0;
	Stopwatch stopwatch;
	StatisticsData statistics;
	double playSpeed = 1.0;
	bool autoplay = false;
	bool episodeChanged = false;

	while (System::Update())
	{
		{//画面右側のファイル関係の描画]
			int32 infoWidth = 300;
			Mat3x2 mat = Mat3x2::Translate(Scene::Width() - infoWidth - 50, 0);
			const Transformer2D transformer{ mat, TransformCursor::Yes };


			if (listBoxState.selectedItemIndex) {
				sedai = listBoxState.items[*listBoxState.selectedItemIndex];
			}
			else {
				sedai = U"?";
			}
			mediumFont(U"参照中のファイルパス").draw(Vec2{ 10, 60 }, Palette::Black);
			Rect pathRect{ 10, 90, infoWidth - 20, 130 };
			tinyFont(path).draw(pathRect, Palette::Black);
			if (SimpleGUI::Button(U"\U000F024B フォルダ選択", Vec2{ 10 , 150 }))
			{
				statistics.reset();
				Optional<FilePath> parent = Dialog::SelectFolder(U"\\\\wsl.localhost\\Ubuntu\\home\\shiba\\repos\\school\\rl\\avoid");
				path = parent.value();
				listBoxState.items.clear();
				if (not parent) {
					throw Error(U"フォルダ選択に失敗しました");
				}
				statistics.reset();
				pathList.clear();
				for (auto path1 : FileSystem::DirectoryContents(parent.value()))
				{
					String path2 = getEpisodeFromString(path1);
					State s = getStateFromCsv(path1);
					statistics.episodeNum++;
					statistics.win.push_back(s.win);
					statistics.totalReward.push_back(s.totalReward);
					path2 += (s.win ? U" (成功)" : U"");
					listBoxState.items << path2;
					pathList << path1;
				}
				statistics.calc();
			}
			episodeChanged = SimpleGUI::ListBox(listBoxState, Vec2{ 10, 200 }, infoWidth - 10 * 2, 540);
		}
		if (episodeChanged)autoplay = true;

		double value = stateT;
		{ //再生に関するUI
			int32 playWidth = 400;
			Mat3x2 mat = Mat3x2::Translate(400, 0);
			const Transformer2D transformer{ mat, TransformCursor::Yes };
			// SimpleGUI::CheckBox(autoplay, U"自動再生", Vec2{ 0, 100 });
			if (SimpleGUI::Button(U"\U000F040B 再生", Vec2{ 0, 205 }, unspecified))autoplay = true;
			if (SimpleGUI::Button(U"\U000F1B7A 停止", Vec2{ 110, 205 }, unspecified))autoplay = false;
			SimpleGUI::Slider(U"\U000F08FF {:.1f}"_fmt(playSpeed), playSpeed, 0.5, 5.0, Vec2{ 220, 205 }, 60, 100);
			SimpleGUI::Slider(U"t = {}"_fmt(value), value, 0.0, (double)(state.myPos.size()) - 1, Vec2{ 0, 160 }, 60, playWidth - 80);

			mediumFont(U"試行回数").draw(Arg::topCenter = Vec2{ playWidth / 2, 50 }, Palette::Black);
			hugeFont(getEpisodeNumberFromString(sedai)).draw(Arg::topCenter = Vec2{ playWidth / 2, 70 }, Palette::Black);

			if (listBoxState.selectedItemIndex) {
				mediumFont(U"このエピソードの結果：{}"_fmt((statistics.win[*listBoxState.selectedItemIndex] ? U"成功" : U"失敗"))).draw(Vec2{ 0, 250 }, Palette::Black);
				// if(!state.myPos.empty())mediumFont(U"デバッグ：{}"_fmt((state.myPos.back().y))).draw(Vec2{0, 00}, Palette::Black);
				double winRate = statistics.winRate[*listBoxState.selectedItemIndex];
				mediumFont(U"精度： {:.2f}%"_fmt(100 * winRate)).draw(Vec2{ 0, 280 }, Palette::Black);
				int32 graphHeight = 300;
				Rect graphRect(0, 330, playWidth, graphHeight);
				graphRect.draw(Palette::Whitesmoke).drawFrame(0, 2, Palette::Black);


				for (auto i : step(4)) {
					graphRect.top().movedBy(0, graphHeight / 4.0 * i).draw(ColorF(0.2));
				}

				Line(graphRect.left().movedBy(playWidth / (double)statistics.episodeNum * (*listBoxState.selectedItemIndex), 0)).draw(Palette::Red);

				for (auto i : step(Max(0, statistics.episodeNum - 1))) {
					float x1 = graphRect.tl().x + (float)graphRect.w / (float)statistics.episodeNum * i;
					float y1 = graphRect.tl().y + (1.0f - statistics.winRate[i]) * graphHeight;
					float x2 = graphRect.tl().x + (float)graphRect.w / (float)statistics.episodeNum * (i + 1);
					float y2 = graphRect.tl().y + (1.0f - statistics.winRate[i + 1]) * graphHeight;
					Line(x1, y1, x2, y2).draw(Palette::Black);
					if (i == statistics.calc_width) {
						Rect(graphRect.tl(), x1-graphRect.tl().x, graphRect.h).draw(ColorF(1.0, 0.5, 0.5, 0.2));
					}
				}
			}
			double v = statistics.width;
			SimpleGUI::Slider(U"平均の幅 {}"_fmt(v), v, 1.0, 1000.0, Vec2{ 0, 650 }, 120, playWidth - 100);
			statistics.width = (int32)v;
			if (SimpleGUI::Button(U"平均再計算", Vec2{ 0, 695 }))
			{
				statistics.calc();
			}

		}
		stateT = (int16)value;
		if (stateT < state.myPos.size()) {
			int32 width = state.width;
			int32 height = state.height;
			{
				Mat3x2 mat = Mat3x2::Scale(10).translated(50, 120);
				//				Mat3x2 mat = Mat3x2::Translate(0, -state.myPos[stateT].y).scaled(10).translated(20, Scene::Height() - 100);
				const Transformer2D transformer{ mat, TransformCursor::Yes };
				Rect(0, 0, width, height).draw(Palette::Lightgray).drawFrame(0, 0.2, Palette::Black);
				Circle(state.myPos[stateT], state.mySize).draw(Palette::Blue);
				for (auto i : step(state.enemyNum)) {
					Circle(state.enemyPos[i][stateT], state.enemySize[i]).draw(Palette::Red);
				}
				for (int16 t = 0; t < stateT; t++) {
					Line(state.myPos[t], state.myPos[t + 1]).draw(0.1, Palette::Black);
				}

			}
			tinyFont(U"({:.1f}, {:.1f})"_fmt(state.myPos[stateT].x, state.myPos[stateT].y)).draw(60, 130, Palette::Black);

			if (autoplay) {
				stateT = (int16)stopwatch.ms() / (100.0 / (playSpeed* playSpeed));
				if (stateT >= state.myPos.size()) {
					stateT = 0;
					stopwatch.restart();
					if (*listBoxState.selectedItemIndex < pathList.size() - 1) (*listBoxState.selectedItemIndex)++;
					episodeChanged = true;
				}
			}
		}
		else {
			Mat3x2 mat = Mat3x2::Scale(10).translated(50, 120);
			//				Mat3x2 mat = Mat3x2::Translate(0, -state.myPos[stateT].y).scaled(10).translated(20, Scene::Height() - 100);
			const Transformer2D transformer{ mat, TransformCursor::Yes };
			Rect(0, 0, 30, 60).draw(Palette::Lightgray).drawFrame(0, 0.2, Palette::Black);
		}

		// 新しいエピソードを選択したら
		if (episodeChanged)
		{
			stopwatch.restart();
			size_t idx = *listBoxState.selectedItemIndex;
			String path = pathList[idx];
			state = getStateFromCsv(path);
		}
	}
}
