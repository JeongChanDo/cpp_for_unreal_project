#include <Windows.h>
#include <iostream>

void GetCursorPosition()
{
    POINT cursorPos;
    if (GetCursorPos(&cursorPos))
    {
        int x = cursorPos.x;
        int y = cursorPos.y;
        // 마우스 커서 위치를 사용하여 원하는 작업 수행
        // 예: 좌표 출력
        std::cout << "Cursor position: x = " << x << " , y = " << y << std::endl;
    }
}

void MoveCursorTo(int x, int y)
{
    SetCursorPos(x, y);
}

int main()
{
    int startX = 80; // 시작 x 좌표
    int startY = 80; // 시작 y 좌표
    int duration = 5; // 프로그램 실행 시간 (초)
    int interval = 500; // 좌표 갱신 간격 (밀리초)
    int step = 20; // x 좌표 증가량

    int targetX = startX;
    int targetY = startY;

    // 일정 시간 동안 주기적으로 x 좌표를 증가시켜 마우스 커서 이동
    for (int i = 0; i < duration * 1000 / interval; i++)
    {
        targetX += step;
        MoveCursorTo(targetX, targetY);
        GetCursorPosition();
        Sleep(interval);
    }

    return 0;
}
