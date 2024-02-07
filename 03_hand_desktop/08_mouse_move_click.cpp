#include <Windows.h>
#include <iostream>


void MoveCursorTo(int x, int y)
{
    SetCursorPos(x, y);
}

void MouseClick(int button)
{
    INPUT input = { 0 };
    input.type = INPUT_MOUSE;
    input.mi.dwFlags = button;

    SendInput(1, &input, sizeof(INPUT));
}

int main()
{
    int startX = 2500; // 시작 x 좌표
    int startY = 80; // 시작 y 좌표
    int duration = 8; // 프로그램 실행 시간 (초)
    int interval = 500; // 좌표 갱신 간격 (밀리초)
    int step = -20; // x 좌표 감소량

    int targetX = startX;
    int targetY = startY;

    // 일정 시간 동안 주기적으로 x 좌표를 감소시켜 마우스 커서 이동 후 클릭
    for (int i = 0; i < duration * 1000 / interval; i++)
    {
        targetX += step;

        MoveCursorTo(targetX, targetY);
        if (i%2 == 0)
            MouseClick(MOUSEEVENTF_RIGHTDOWN);
        else
            MouseClick(MOUSEEVENTF_RIGHTUP);
        Sleep(interval);
    }

    return 0;
}
