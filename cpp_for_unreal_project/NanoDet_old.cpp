#include "NanoDet.h"

CV2NanoDetONNX::CV2NanoDetONNX(
    std::string model_path = "nanodet_m.onnx",
    int input_shape = 320,
    float class_score_th = 0.35f,
    float nms_th = 0.6f
)
{
    this->input_shape = cv::Size(input_shape, input_shape);
    this->class_score_th = class_score_th;
    this->nms_th = nms_th;

    this->net = cv::dnn::readNetFromONNX(model_path);
    this->output_names = {
         "cls_pred_stride_8",
         "dis_pred_stride_8",
         "cls_pred_stride_16",
         "dis_pred_stride_16",
         "cls_pred_stride_32",
         "dis_pred_stride_32"
    };



    for (int index = 0; index < 3; index++)
    {
        cv::Size grid_size(this->input_shape.width / STRIDES[index], this->input_shape.height / STRIDES[index]);
        cv::Mat grid_point = make_grid_point(grid_size, STRIDES[index]);
        grid_points.push_back(grid_point);
    }


    PROJECT = cv::Mat(8, 1, CV_32FC1, data);
}


cv::Mat CV2NanoDetONNX::inference(cv::Mat image) {
    cv::Mat temp_image = image.clone();
    int image_height = image.rows;
    int image_width = image.cols;

    cv::Mat resized_image;
    int new_height, new_width, top, left;
    resize_image(temp_image, true, resized_image, new_height, new_width, top, left);


    cv::Mat blob = preProcess(resized_image);

    net.setInput(blob);
    std::vector<cv::Mat> preds;

    net.forward(preds, output_names);


    std::cout << preds.size() << std::endl;


    std::vector<cv::Mat> bboxes, scores, class_ids;
    std::cout << std::endl << std::endl << std::endl;
    postProcess(preds, bboxes, scores, class_ids);


    /*

    float ratio_height = (float)image_height / (float)new_height;
    float ratio_width = (float)image_width / (float)new_width;
    for (int i = 0; i < bboxes.rows; i++) {
        bboxes.at<float>(i, 0) = std::max(int((bboxes.at<float>(i, 0) - left) * ratio_width), 0);
        bboxes.at<float>(i, 1) = std::max(int((bboxes.at<float>(i, 1) - top) * ratio_height), 0);
        bboxes.at<float>(i, 2) = std::min(int((bboxes.at<float>(i, 2) - left) * ratio_width), image_width);
        bboxes.at<float>(i, 3) = std::min(int((bboxes.at<float>(i, 3) - top) * ratio_height), image_height);
    }
    return bboxes, scores, class_ids;
    */
    return resized_image;
}


cv::Mat CV2NanoDetONNX::make_grid_point(cv::Size grid_size, int stride) {
    int grid_width = grid_size.width;
    int grid_height = grid_size.height;

    std::vector<int> shift_x(grid_width);
    std::vector<int> shift_y(grid_height);

    for (int i = 0; i < grid_width; ++i) {
        shift_x[i] = i * stride;
    }

    for (int i = 0; i < grid_height; ++i) {
        shift_y[i] = i * stride;
    }

    cv::Mat grid_point(grid_height * grid_width, 2, CV_32F);
    int index = 0;
    for (int y = 0; y < grid_height; ++y) {
        for (int x = 0; x < grid_width; ++x) {
            float cx = shift_x[x] + 0.5 * (stride - 1);
            float cy = shift_y[y] + 0.5 * (stride - 1);
            grid_point.at<float>(index, 0) = cx;
            grid_point.at<float>(index, 1) = cy;
            index++;
        }
    }
    return grid_point;
}



void CV2NanoDetONNX::resize_image(cv::Mat image, bool keep_ratio, cv::Mat& resized_image, int& new_height, int& new_width, int& top, int& left) {


    if (keep_ratio && image.rows != image.cols) {
        float hw_scale = static_cast<float>(image.rows) / image.cols;
        if (hw_scale > 1) {
            new_height = this->input_shape.height;
            new_width = static_cast<int>(this->input_shape.width / hw_scale);

            cv::resize(image, resized_image, cv::Size(new_width, new_height), cv::INTER_AREA);

            left = static_cast<int>((this->input_shape.width - new_width) * 0.5);

            cv::copyMakeBorder(resized_image, resized_image, 0, 0, left, this->input_shape.width - new_width - left, cv::BORDER_CONSTANT, cv::Scalar(0));

        }
        else {
            new_height = static_cast<int>(this->input_shape.height * hw_scale);
            new_width = this->input_shape.width;

            cv::resize(image, resized_image, cv::Size(new_width, new_height), cv::INTER_AREA);

            top = static_cast<int>((this->input_shape.height - new_height) * 0.5);

            cv::copyMakeBorder(resized_image, resized_image, top, this->input_shape.height - new_height - top, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0));
        }
    }
    else
    {
        cv::resize(image, resized_image, this->input_shape, 0, 0, cv::INTER_AREA);
    }
}


cv::Mat CV2NanoDetONNX::preProcess(cv::Mat image) {
    cv::Mat processedImage;

    image.convertTo(processedImage, CV_32F);



    std::vector<cv::Mat> channels;
    cv::split(processedImage, channels);  // frame의 각 채널 분리

    // 각 채널에 대해 연산 수행
    for (int i = 0; i < 3; i++) {
        channels[i] = channels[i] - mean[i];
        channels[i] = channels[i] / std[i];
    }
    cv::merge(channels, processedImage);  // 채널 다시 결합


    processedImage = cv::dnn::blobFromImage(processedImage, 1.0, processedImage.size(), 0, false, false, CV_32F);
    std::cout << processedImage.size[0] << std::endl;
    std::cout << processedImage.size[1] << std::endl;
    std::cout << processedImage.size[2] << std::endl;
    std::cout << processedImage.size[3] << std::endl;

    return processedImage;
}


void CV2NanoDetONNX::postProcess(const std::vector<cv::Mat>& predict_results, std::vector<cv::Mat>& class_scores, std::vector<cv::Mat>& bbox_predicts, std::vector<cv::Mat>& class_ids) {
    for (int i = 0; i < predict_results.size(); i += 2) {
        
        //std::cout << "predict_results[" << i <<"] : " << predict_results[i].size[0] << " X " << predict_results[i].size[1] << " X " << predict_results[i].size[2] << std::endl;
        //std::cout << "predict_results[" << i + 1 << "] : " << predict_results[i + 1].size[0] << " X " << predict_results[i + 1].size[1] << " X " << predict_results[i + 1].size[2] << std::endl;


        class_scores.push_back(predict_results[i]);
        bbox_predicts.push_back(predict_results[i + 1]);
    }

    /*
    std::cout << "class scores size : " << class_scores.size() << std::endl;
    std::cout << "bbox_predicts size : " << bbox_predicts.size() << std::endl;
    std::cout << "class_ids size : " << class_ids.size() << std::endl;
    std::cout << std::endl << std::endl << std::endl;
    */

    get_bboxes_single(
        class_scores,
        bbox_predicts,
        1,
        false,
        1000
    );

}


void CV2NanoDetONNX::get_bboxes_single(
    const std::vector<cv::Mat>& class_scores,
    const std::vector<cv::Mat>& bbox_predicts,
    float scale_factor,
    bool rescale,
    int topk
) {
    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;

    for (int i = 0; i < class_scores.size(); i++) {
        cv::Mat class_score = class_scores[i];
        cv::Mat bbox_predict = bbox_predicts[i];

        cv::Mat grid_point = grid_points[i];
        int stride = STRIDES[i];

        /*
        std::cout << "before class_score size : " << class_score.size() << std::endl;
        std::cout << "class_score size[0] : " << class_score.size[0] << std::endl;
        std::cout << "class_score size[1] : " << class_score.size[1] << std::endl;
        std::cout << "class_score size[2] : " << class_score.size[2] << std::endl;
        std::cout << "class_score dims : " << class_score.dims << std::endl;
        std::cout << "rows : " << class_score.rows << ", cols : " << class_score.cols << std::endl << std::endl;

        std::cout << "bbox_predict size : " << bbox_predict.size() << std::endl;
        std::cout << "bbox_predict size[0] : " << bbox_predict.size[0] << std::endl;
        std::cout << "bbox_predict size[1] : " << bbox_predict.size[1] << std::endl;
        std::cout << "bbox_predict size[2] : " << bbox_predict.size[2] << std::endl;
        std::cout << "bbox_predict dims : " << bbox_predict.dims << std::endl << std::endl;

        std::cout << "rows : " << bbox_predict.rows << ", cols : " << bbox_predict.cols << std::endl;
        */
     
        // 1 x 1600 x 1 -> 1 col x 1600 row
        if (class_score.dims == 3) {
            class_score = class_score.reshape(1, class_score.size[1]);
        }
        // 1 x 1600 x 32 -> 32 col x 1600 row
        if (bbox_predict.dims == 3) {
            bbox_predict = bbox_predict.reshape(1, bbox_predict.size[1]);
        }
        /*
        std::cout << class_score.row(0) << std::endl;
        std::cout << bbox_predict.row(0) << std::endl;

        std::cout << std::endl << "after class_score size : " << class_score.size() << std::endl;
        std::cout << "rows : " << class_score.rows << ", cols : " << class_score.cols << std::endl;
        std::cout << "class_score dims : " << class_score.dims << std::endl;

        std::cout << "bbox_predict size : " << bbox_predict.size() << std::endl;
        std::cout << "rows : " << bbox_predict.rows << ", cols : " << bbox_predict.cols << std::endl;
        std::cout << "bbox_predict dims : " << bbox_predict.dims << std::endl;
        */


        bbox_predict = bbox_predict.reshape(1, REG_MAX + 1);
        /*
        std::cout << bbox_predict.col(0) << std::endl;
        std::cout << "bbox_predict reshape size : " << bbox_predict.size() << std::endl;
        std::cout << "rows : " << bbox_predict.rows << ", cols : " << bbox_predict.cols << std::endl;
        */

        bbox_predict = softmax(bbox_predict, 0).t();
        /*

        std::cout << "after softmax bbox_predict size : " << bbox_predict.size() << std::endl;
        std::cout << "rows : " << bbox_predict.rows << ", cols : " << bbox_predict.cols << std::endl;
        std::cout << "PROJECT rows : " << PROJECT.rows << ", PROJECT cols : " << PROJECT.cols << std::endl;
        */


        /*
        std::vector<float> bbox_predict_vec;
        for (int i = 0; i < bbox_predict.rows ; i++)
        {
            std::cout << bbox_predict.row(i) << std::endl;
            if (i == 3) break;
            cv::Mat tmp = bbox_predict.row(i);
            std::cout << "tmp rows : " << tmp.rows << ", tmp cols : " << tmp.cols << std::endl;

            std::cout << tmp.dot(PROJECT) << std::endl;

            //bbox_predict_vec.push_back(dot_res);
        }
        std::cout << " bbox_predict_vec size : " << bbox_predict_vec.size() << std::endl;
        */

        //bbox_predict = bbox_predict.dot(PROJECT);
        bbox_predict = bbox_predict * PROJECT;

        //cv::Mat tmp = bbox_predict * PROJECT.size();
        //std::cout << "dot operation  bbox_predict size : " << bbox_predict.size() << std::endl;
        //std::cout << "bbox_predict rows : " << bbox_predict.rows << ", bbox_predict cols : " << bbox_predict.cols << std::endl;


        bbox_predict = bbox_predict.reshape(1, 4).t();

        //std::cout << "reshape bbox_predict size : " << bbox_predict.size() << std::endl << std::endl;
        //std::cout << "bbox_predict rows : " << bbox_predict.rows << ", bbox_predict cols : " << bbox_predict.cols << ", dims : " << bbox_predict.dims << std::endl;


        bbox_predict = bbox_predict * stride;

        if (topk > 0 && topk < class_score.size[0]) {
            cv::Mat max_scores;
            std::cout << class_score.size() << std::endl;
            cv::reduce(class_score, max_scores, 1, cv::REDUCE_MAX);
            std::cout << max_scores.rowRange(0, 10) << std::endl;

            cv::Mat sorted_indexes;
            cv::sortIdx(max_scores, sorted_indexes, cv::SORT_EVERY_COLUMN + cv::SORT_DESCENDING);
            std::cout << sorted_indexes.rowRange(0, 10) << std::endl;


            cv::Mat topk_indexes = sorted_indexes.rowRange(0, topk);

            std::cout << topk_indexes.rowRange(0, 10) << std::endl;
            std::cout << topk_indexes.size() << std::endl;

            /*
            grid_point = grid_point.rowRange(topk_indexes, topk_indexes + 1);
            */
            std::cout << "grid_point, bbox_predict, class_score" << std::endl;
            std::cout << grid_point.size() << std::endl;
            std::cout << bbox_predict.size() << std::endl;
            std::cout << class_score.size() << std::endl;

            std::cout << "gridpoint" << std::endl;
            std::cout << grid_point.rowRange(0, 10) << std::endl;

            cv::Mat grid_point_sorted = cv::Mat(topk_indexes.rows, grid_point.cols, CV_32F);
            cv::Mat bbox_predict_sorted = cv::Mat(topk_indexes.rows, bbox_predict.cols, CV_32F);
            cv::Mat class_score_sorted = cv::Mat(topk_indexes.rows, class_score.cols, CV_32F);

            for (int i = 0; i < topk_indexes.rows; i++) {
                grid_point_sorted.at<float>(i, 0) = grid_point.at<float>(topk_indexes.at<int>(0, i), 0);
                grid_point_sorted.at<float>(i, 1) = grid_point.at<float>(topk_indexes.at<int>(0, i), 1);

                bbox_predict_sorted.at<float>(i, 0) = bbox_predict.at<float>(topk_indexes.at<int>(0, i), 0);
                bbox_predict_sorted.at<float>(i, 1) = bbox_predict.at<float>(topk_indexes.at<int>(0, i), 1);
                bbox_predict_sorted.at<float>(i, 2) = bbox_predict.at<float>(topk_indexes.at<int>(0, i), 2);
                bbox_predict_sorted.at<float>(i, 3) = bbox_predict.at<float>(topk_indexes.at<int>(0, i), 3);

                class_score_sorted.at<float>(i, 0) = class_score.at<float>(topk_indexes.at<int>(0, i), 0);
            }
            grid_point = grid_point_sorted;
            bbox_predict = bbox_predict_sorted;
            class_score = class_score_sorted;

            std::cout << "sorted gridpoint" << std::endl;
            std::cout << grid_point.rowRange(0, 5) << std::endl;
            std::cout << "sorted bbox_predict" << std::endl;
            std::cout << bbox_predict.rowRange(0, 5) << std::endl;
            std::cout << "sorted class_score" << std::endl;
            std::cout << class_score.rowRange(0, 5) << std::endl;
            std::cout << "grid_point, bbox_predict, class_score" << std::endl;
            std::cout << grid_point.size() << std::endl;
            std::cout << bbox_predict.size() << std::endl;
            std::cout << class_score.size() << std::endl;

        }

        /*
        cv::Mat x1 = grid_point.col(0) - bbox_predict.col(0);
        cv::Mat y1 = grid_point.col(1) - bbox_predict.col(1);
        cv::Mat x2 = grid_point.col(0) + bbox_predict.col(2);
        cv::Mat y2 = grid_point.col(1) + bbox_predict.col(3);
        */
    }
    std::cout << std::endl << std::endl;

}


cv::Mat CV2NanoDetONNX::softmax(cv::Mat x, int axis)
{
    cv::Mat x_exp, x_sum, s;
    cv::exp(x, x_exp);
    //std::cout << "x_exp size : " << x_exp.size() << std::endl;
    
    cv::reduce(x_exp, x_sum, axis, cv::REDUCE_SUM, CV_32F);
    //std::cout << "x_sum size : " << x_sum.size() << std::endl;




    cv::Mat expanded_x_sum;
    cv::repeat(x_sum, 8, 1, expanded_x_sum);
    //std::cout << "exp x sum size : " << expanded_x_sum.size() << std::endl;

    cv::divide(x_exp, expanded_x_sum, s);

    /*
    std::cout << "expanded_x_sum 0번째 행 값: " << expanded_x_sum.col(0) << std::endl;
    std::cout << "x_exp 0번째 행 값: " << x_exp.col(0) << std::endl;
    std::cout << "s 0번째 행 값: " << s.col(0) << std::endl;
    */
    return s;
    //s = x_exp / x_sum;
    //return s;
}























std::vector<cv::Rect> CV2NanoDetONNX::getColorFilteredBoxes(cv::Mat image) {
    // 살색 영역의 범위 지정 (HSV 색 공간)
    cv::Scalar lower_skin(0, 20, 70);
    cv::Scalar upper_skin(20, 255, 255);

    // 이미지를 HSV 색 공간으로 변환
    cv::Mat hsv_image;
    cv::cvtColor(image, hsv_image, cv::COLOR_BGR2HSV);

    // 살색 영역을 마스크로 만들기
    cv::Mat skin_mask;
    cv::inRange(hsv_image, lower_skin, upper_skin, skin_mask);

    // 모폴로지 연산을 위한 구조 요소 생성
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));

    // 모폴로지 열림 연산 적용
    cv::morphologyEx(skin_mask, skin_mask, cv::MORPH_OPEN, kernel);

    // 마스크를 이용하여 살색 영역 추출
    cv::Mat skin_image;
    cv::bitwise_and(image, image, skin_image, skin_mask);

    // 살색 영역에 대한 바운딩 박스 추출
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(skin_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Rect> bounding_boxes;
    for (const auto& contour : contours) {
        cv::Rect bounding_box = cv::boundingRect(contour);
        // 크기가 작은 박스와 큰 박스 제거
        if (bounding_box.width * bounding_box.height > 100 * 100) {
            // 약간 박스 더 크게
            bounding_box.x -= 10;
            bounding_box.y -= 10;
            bounding_box.width += 20;
            bounding_box.height += 20;
            bounding_boxes.push_back(bounding_box);
        }
    }

    return bounding_boxes;
}


cv::Mat CV2NanoDetONNX::draw_debug_roi(cv::Mat image, std::vector<cv::Rect> bboxes, std::vector<float> scores, std::vector<int> class_ids, int x, int y) {
    for (int i = 0; i < bboxes.size(); i++) {
        cv::Rect bbox = bboxes[i];
        int x1 = bbox.x;
        int y1 = bbox.y;
        int x2 = bbox.x + bbox.width;
        int y2 = bbox.y + bbox.height;

        cv::rectangle(
            image,
            cv::Point(x1 + x, y1 + y),
            cv::Point(x2 + x, y2 + y),
            cv::Scalar(0, 255, 0),
            2
        );

        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << scores[i];
        std::string score = ss.str();
        std::string text = std::to_string(class_ids[i]) + ":" + score;
        cv::putText(
            image,
            text,
            cv::Point(bbox.x + x, bbox.y - 10 + y),
            cv::FONT_HERSHEY_SIMPLEX,
            0.7,
            cv::Scalar(0, 255, 0),
            2
        );
    }

    return image;
}


