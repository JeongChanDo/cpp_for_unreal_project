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
        std::cout << "Cursor position: x = "<< x << " , y = " << y << std::endl;
    }
}

int main()
{
    int duration = 5; // 프로그램 실행 시간 (초)
    int interval = 500; // 마우스 커서 위치 갱신 간격 (밀리초)

    // 일정 시간 동안 주기적으로 마우스 커서 위치 얻기
    for (int i = 0; i < duration * 1000 / interval; i++)
    {
        GetCursorPosition();
        Sleep(interval);
    }

    return 0;
}
